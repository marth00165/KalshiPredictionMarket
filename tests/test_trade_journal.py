import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from app.bot import AdvancedTradingBot


def test_trade_journal_created_and_appended(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    bot = AdvancedTradingBot.__new__(AdvancedTradingBot)
    bot.config = MagicMock()
    bot.config.is_dry_run = True
    bot.cycle_count = 1

    bot._ensure_daily_trade_journal()

    date_key = datetime.utcnow().strftime("%Y-%m-%d")
    journal_path = Path("reports") / "trade_journal" / f"{date_key}.json"
    assert journal_path.exists()

    with journal_path.open("r") as f:
        payload = json.load(f)
    assert payload["date_utc"] == date_key
    assert isinstance(payload["trades"], list)
    assert payload["trades"] == []

    bot._append_trade_journal_entry(
        {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "cycle_id": 1,
            "dry_run": True,
            "market_id": "KXTEST-1",
            "action": "buy_yes",
            "reasoning": "Value edge from model analysis",
        }
    )

    with journal_path.open("r") as f:
        payload = json.load(f)

    assert payload["trade_count"] == 1
    assert payload["trades"][0]["market_id"] == "KXTEST-1"
    assert payload["trades"][0]["reasoning"] == "Value edge from model analysis"

