import pytest

from app.storage.db import DatabaseManager


@pytest.mark.asyncio
async def test_analysis_reasoning_persist_and_query(tmp_path):
    db = DatabaseManager(str(tmp_path / "reasoning.sqlite"))
    await db.initialize()

    written = await db.save_analysis_reasoning_entries(
        [
            {
                "cycle_id": 1,
                "market_id": "MKT-1",
                "event_ticker": "EVT-1",
                "series_ticker": "SER-A",
                "platform": "kalshi",
                "provider": "openai",
                "reasoning": "First blurb",
                "edge": 0.12,
                "confidence": 0.75,
                "analyzed_at_utc": "2026-02-18T10:00:00Z",
            },
            {
                "cycle_id": 1,
                "market_id": "MKT-2",
                "event_ticker": "EVT-2",
                "series_ticker": "SER-B",
                "platform": "kalshi",
                "provider": "openai",
                "reasoning": "Second blurb",
                "edge": 0.20,
                "confidence": 0.90,
                "analyzed_at_utc": "2026-02-18T11:00:00Z",
            },
        ]
    )
    assert written == 2

    latest = await db.get_recent_analysis_reasoning(limit=1)
    assert len(latest) == 1
    assert latest[0]["market_id"] == "MKT-2"
    assert latest[0]["reasoning"] == "Second blurb"

    filtered = await db.get_recent_analysis_reasoning(
        limit=5,
        platform="kalshi",
        series_tickers=["SER-A"],
    )
    assert len(filtered) == 1
    assert filtered[0]["market_id"] == "MKT-1"


@pytest.mark.asyncio
async def test_analysis_reasoning_upsert_same_cycle_market(tmp_path):
    db = DatabaseManager(str(tmp_path / "reasoning_upsert.sqlite"))
    await db.initialize()

    await db.save_analysis_reasoning_entries(
        [
            {
                "cycle_id": 7,
                "market_id": "MKT-X",
                "platform": "kalshi",
                "provider": "openai",
                "reasoning": "Old",
                "analyzed_at_utc": "2026-02-18T10:00:00Z",
            }
        ]
    )
    await db.save_analysis_reasoning_entries(
        [
            {
                "cycle_id": 7,
                "market_id": "MKT-X",
                "platform": "kalshi",
                "provider": "openai",
                "reasoning": "Updated",
                "analyzed_at_utc": "2026-02-18T10:05:00Z",
            }
        ]
    )

    rows = await db.get_recent_analysis_reasoning(limit=5)
    assert len(rows) == 1
    assert rows[0]["reasoning"] == "Updated"
