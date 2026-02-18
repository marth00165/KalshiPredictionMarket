from unittest.mock import MagicMock

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
