import sqlite3
import aiosqlite
import logging
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages SQLite database for market snapshots and bot status.

    Supports:
    - WAL mode for better concurrency
    - Simple schema migrations
    - Idempotent snapshot inserts
    - Status/heartbeat tracking
    """

    def __init__(self, db_path: str = "kalshi.sqlite"):
        self.db_path = db_path
        self.initialized = False

    async def initialize(self):
        """Initialize database, enable WAL, and run migrations."""
        if self.initialized:
            return

        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute("PRAGMA synchronous=NORMAL;")

            # Create schema version table if it doesn't exist
            await db.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
            """)

            await self._run_migrations(db)

        self.initialized = True
        logger.info(f"✅ Database initialized at {self.db_path}")

    def connect(self):
        """Get an aiosqlite connection context manager."""
        return aiosqlite.connect(self.db_path, timeout=30.0)

    async def _run_migrations(self, db: aiosqlite.Connection):
        """Run pending schema migrations."""
        # Current schema version
        async with db.execute("SELECT MAX(version) FROM schema_version") as cursor:
            row = await cursor.fetchone()
            current_version = row[0] if row and row[0] is not None else 0

        # Define migrations
        # Version 1: Initial schema
        # Version 2: Bankroll, Positions, Executions
        migrations = [
            # Version 1
            """
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                snapshot_hour_utc TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                UNIQUE(market_id, platform, snapshot_hour_utc)
            );
            CREATE TABLE IF NOT EXISTS status (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at_utc TEXT NOT NULL
            );
            """,
            # Version 2
            """
            CREATE TABLE IF NOT EXISTS bankroll_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                balance REAL NOT NULL,
                change REAL NOT NULL,
                reason TEXT NOT NULL,
                reference_id TEXT,
                created_at_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                cost REAL NOT NULL,
                status TEXT NOT NULL,
                external_order_id TEXT,
                opened_at_utc TEXT NOT NULL,
                closed_at_utc TEXT,
                created_at_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                cost REAL NOT NULL,
                external_order_id TEXT,
                status TEXT NOT NULL,
                executed_at_utc TEXT NOT NULL,
                created_at_utc TEXT NOT NULL
            );
            """,
            # Version 3: Add result column to positions
            """
            ALTER TABLE positions ADD COLUMN result REAL DEFAULT 0.0;
            """,
            # Version 4: Add pnl_history table
            """
            CREATE TABLE IF NOT EXISTS pnl_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                profit REAL NOT NULL,
                balance REAL NOT NULL,
                created_at_utc TEXT NOT NULL
            );
            """,
            # Version 5: Add client_order_id and cycle_id to executions
            """
            ALTER TABLE executions ADD COLUMN client_order_id TEXT;
            ALTER TABLE executions ADD COLUMN cycle_id INTEGER;
            CREATE INDEX IF NOT EXISTS idx_executions_client_order_id ON executions(client_order_id);
            """,
            # Version 6: Add filled_quantity, avg_fill_price, and remaining_quantity to executions
            """
            ALTER TABLE executions ADD COLUMN filled_quantity REAL DEFAULT 0.0;
            ALTER TABLE executions ADD COLUMN avg_fill_price REAL DEFAULT 0.0;
            ALTER TABLE executions ADD COLUMN remaining_quantity REAL DEFAULT 0.0;
            """,
            # Version 7: Add reconciliation tracking to executions
            """
            ALTER TABLE executions ADD COLUMN reconcile_attempts INTEGER DEFAULT 0;
            ALTER TABLE executions ADD COLUMN last_reconciled_at_utc TEXT;
            """,
            # Version 8: Persist LLM analysis blurbs for future prompt context
            """
            CREATE TABLE IF NOT EXISTS analysis_reasoning (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id INTEGER,
                market_id TEXT NOT NULL,
                event_ticker TEXT,
                series_ticker TEXT,
                platform TEXT NOT NULL,
                title TEXT,
                selection TEXT,
                yes_price REAL,
                fair_probability REAL,
                edge REAL,
                confidence REAL,
                action TEXT,
                position_size REAL,
                provider TEXT NOT NULL,
                reasoning TEXT NOT NULL,
                analyzed_at_utc TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                UNIQUE(cycle_id, market_id, platform, provider)
            );
            CREATE INDEX IF NOT EXISTS idx_analysis_reasoning_analyzed_at
            ON analysis_reasoning(analyzed_at_utc DESC);
            CREATE INDEX IF NOT EXISTS idx_analysis_reasoning_series
            ON analysis_reasoning(platform, series_ticker, analyzed_at_utc DESC);
            """
        ]

        for i, sql in enumerate(migrations):
            version = i + 1
            if version > current_version:
                logger.info(f"Applying migration version {version}...")
                try:
                    await db.executescript(sql)
                    await db.execute(
                        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                        (version, datetime.utcnow().isoformat())
                    )
                    await db.commit()
                    logger.info(f"✅ Applied migration version {version}")
                except Exception as e:
                    logger.error(f"❌ Failed to apply migration version {version}: {e}")
                    raise

    async def save_market_snapshots(self, markets: List[Dict[str, Any]]):
        """
        Save a batch of market snapshots.

        Args:
            markets: List of market data dictionaries
        """
        # Snapshots are grouped by hour to avoid duplicates from frequent runs
        snapshot_hour = datetime.utcnow().strftime("%Y-%m-%dT%H:00:00Z")
        now = datetime.utcnow().isoformat()

        async with self.connect() as db:
            count = 0
            for market in markets:
                try:
                    await db.execute("""
                        INSERT OR IGNORE INTO market_snapshots
                        (market_id, platform, snapshot_hour_utc, payload_json, created_at_utc)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        market.get('market_id'),
                        market.get('platform'),
                        snapshot_hour,
                        json.dumps(market),
                        now + "Z" if not now.endswith("Z") else now
                    ))
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to save snapshot for {market.get('market_id')}: {e}")

            await db.commit()
            logger.debug(f"Saved {count} market snapshots for {snapshot_hour}")

    async def update_status(self, key: str, value: Any):
        """
        Update a status value in the status table.

        Args:
            key: Status key
            value: Status value (will be converted to string)
        """
        now = datetime.utcnow().isoformat() + "Z"
        async with self.connect() as db:
            await db.execute("""
                INSERT OR REPLACE INTO status (key, value, updated_at_utc)
                VALUES (?, ?, ?)
            """, (key, str(value), now))
            await db.commit()

    async def get_last_status(self) -> Dict[str, Any]:
        """Get all status values as a dictionary."""
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT key, value FROM status") as cursor:
                rows = await cursor.fetchall()
                return {row['key']: row['value'] for row in rows}

    async def save_analysis_reasoning_entries(self, entries: List[Dict[str, Any]]) -> int:
        """
        Persist LLM analysis reasoning entries for later context reuse.

        Returns:
            Number of entries written (inserted or updated).
        """
        if not entries:
            return 0

        now = datetime.utcnow().isoformat() + "Z"
        written = 0

        async with self.connect() as db:
            for entry in entries:
                try:
                    await db.execute(
                        """
                        INSERT INTO analysis_reasoning (
                            cycle_id, market_id, event_ticker, series_ticker, platform,
                            title, selection, yes_price, fair_probability, edge, confidence,
                            action, position_size, provider, reasoning, analyzed_at_utc, created_at_utc
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(cycle_id, market_id, platform, provider) DO UPDATE SET
                            event_ticker=excluded.event_ticker,
                            series_ticker=excluded.series_ticker,
                            title=excluded.title,
                            selection=excluded.selection,
                            yes_price=excluded.yes_price,
                            fair_probability=excluded.fair_probability,
                            edge=excluded.edge,
                            confidence=excluded.confidence,
                            action=excluded.action,
                            position_size=excluded.position_size,
                            reasoning=excluded.reasoning,
                            analyzed_at_utc=excluded.analyzed_at_utc
                        """,
                        (
                            entry.get("cycle_id"),
                            entry.get("market_id"),
                            entry.get("event_ticker"),
                            entry.get("series_ticker"),
                            entry.get("platform"),
                            entry.get("title"),
                            entry.get("selection"),
                            entry.get("yes_price"),
                            entry.get("fair_probability"),
                            entry.get("edge"),
                            entry.get("confidence"),
                            entry.get("action"),
                            entry.get("position_size"),
                            entry.get("provider") or "unknown",
                            entry.get("reasoning") or "",
                            entry.get("analyzed_at_utc") or now,
                            now,
                        ),
                    )
                    written += 1
                except Exception as e:
                    logger.warning(
                        "Failed to persist analysis reasoning for %s: %s",
                        entry.get("market_id"),
                        e,
                    )
            await db.commit()

        return written

    async def get_recent_analysis_reasoning(
        self,
        limit: int = 20,
        platform: Optional[str] = None,
        series_tickers: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load recent analysis blurbs for use as prompt context.
        """
        safe_limit = max(1, int(limit or 1))
        where_clauses: List[str] = []
        params: List[Any] = []

        if platform:
            where_clauses.append("platform = ?")
            params.append(platform)

        if series_tickers:
            normalized_series = [s for s in series_tickers if s]
            if normalized_series:
                placeholders = ",".join(["?"] * len(normalized_series))
                where_clauses.append(f"series_ticker IN ({placeholders})")
                params.extend(normalized_series)

        query = """
            SELECT
                cycle_id, market_id, event_ticker, series_ticker, platform,
                title, selection, yes_price, fair_probability, edge, confidence,
                action, position_size, provider, reasoning, analyzed_at_utc
            FROM analysis_reasoning
        """
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        query += " ORDER BY analyzed_at_utc DESC LIMIT ?"
        params.append(safe_limit)

        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, tuple(params)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def backup(self, backup_dir: str = "backups") -> str:
        """
        Create a backup of the database using SQLite's backup API.

        Args:
            backup_dir: Directory where the backup will be stored.

        Returns:
            The path to the created backup file.
        """
        db_path = Path(self.db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

        backup_folder = Path(backup_dir)
        backup_folder.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{db_path.stem}_{timestamp}{db_path.suffix}"
        backup_path = backup_folder / backup_name

        def _do_sqlite_backup():
            with sqlite3.connect(self.db_path) as src:
                with sqlite3.connect(backup_path) as dst:
                    src.backup(dst)

        # Run the synchronous backup in a thread to avoid blocking the event loop
        await asyncio.to_thread(_do_sqlite_backup)

        logger.info(f"✅ Database backup created: {backup_path}")
        return str(backup_path)
