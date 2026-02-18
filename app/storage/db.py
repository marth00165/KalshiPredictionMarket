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
        return aiosqlite.connect(self.db_path)

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

        async with aiosqlite.connect(self.db_path) as db:
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
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO status (key, value, updated_at_utc)
                VALUES (?, ?, ?)
            """, (key, str(value), now))
            await db.commit()

    async def get_last_status(self) -> Dict[str, Any]:
        """Get all status values as a dictionary."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT key, value FROM status") as cursor:
                rows = await cursor.fetchall()
                return {row['key']: row['value'] for row in rows}

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
