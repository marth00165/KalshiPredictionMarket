from unittest.mock import MagicMock
import json

from app.bot import AdvancedTradingBot


def test_dry_run_analysis_table_includes_reasoning_blurb(capsys):
    bot = AdvancedTradingBot.__new__(AdvancedTradingBot)
    bot.config = MagicMock()
    bot.config.is_dry_run = True

    report = {
        "markets": [
            {
                "market_id": "KXNBAGAME-26FEB19DETNYK-DET",
                "title": "Detroit at New York Winner?",
                "yes_option": "Detroit",
                "prices": {"yes": 0.40},
                "analysis": {
                    "effective_probability": 0.55,
                    "effective_edge": 0.15,
                    "effective_confidence": 0.75,
                    "reasoning": "Model likes Detroit due to matchup and market underpricing.",
                },
                "signal": {
                    "action": "buy_yes",
                    "position_size": 150.0,
                    "reasoning": "Signal confirmed: edge and confidence pass thresholds.",
                },
            }
        ]
    }

    bot._print_dry_run_analysis_table(report)
    output = capsys.readouterr().out

    assert "DRY RUN ANALYSIS RESULTS" in output
    assert "Why" in output
    assert "Signal confirmed: edge and confidence pass thresholds." in output


def test_dry_run_analysis_table_writes_json_in_single_run_mode(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bot = AdvancedTradingBot.__new__(AdvancedTradingBot)
    bot.config = MagicMock()
    bot.config.is_dry_run = True
    bot.single_run_dry_output_json = True
    bot.cycle_count = 3

    report = {
        "markets": [
            {
                "market_id": "KXNBAGAME-26FEB19DETNYK-DET",
                "title": "Detroit at New York Winner?",
                "yes_option": "Detroit",
                "prices": {"yes": 0.40},
                "analysis": {
                    "effective_probability": 0.55,
                    "effective_edge": 0.15,
                    "effective_confidence": 0.75,
                    "reasoning": "Model likes Detroit due to matchup and market underpricing.",
                },
                "signal": {
                    "action": "buy_yes",
                    "position_size": 150.0,
                    "reasoning": "Signal confirmed: edge and confidence pass thresholds.",
                },
            }
        ]
    }

    bot._print_dry_run_analysis_table(report)
    output = capsys.readouterr().out

    assert output == ""
    out_files = sorted((tmp_path / "reports" / "dry_run_analysis").glob("cycle_*.json"))
    assert len(out_files) == 1

    payload = json.loads(out_files[0].read_text())
    assert payload["cycle"] == 3
    assert payload["results_count"] == 1
    assert payload["results"][0]["market_id"] == "KXNBAGAME-26FEB19DETNYK-DET"
    assert payload["results"][0]["reasoning"] == "Signal confirmed: edge and confidence pass thresholds."
