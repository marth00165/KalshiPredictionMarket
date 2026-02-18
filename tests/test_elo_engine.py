import json

from app.analytics import EloEngine
from app.utils.response_parser import ClaudeResponseParser


def _write_sample_games_csv(path):
    path.write_text(
        "\n".join(
            [
                "gameId,gameDateTimeEst,hometeamId,awayteamId,homeScore,awayScore,gameType",
                "1,2024-01-01 19:00:00,1610612737,1610612738,110,100,Regular Season",
                "2,2024-01-02 19:00:00,1610612738,1610612737,120,100,Regular Season",
                "3,2024-01-03 19:00:00,1610612737,1610612738,95,99,Preseason",
            ]
        )
        + "\n"
    )


def test_elo_engine_rebuild_exports_and_probabilities(tmp_path):
    data_path = tmp_path / "kaggleGameData.csv"
    out_path = tmp_path / "elo_ratings.json"
    _write_sample_games_csv(data_path)

    engine = EloEngine(
        data_path=str(data_path),
        output_path=str(out_path),
        initial_rating=1300.0,
        home_advantage=100.0,
        k_factor=20.0,
    )

    ratings = engine.rebuild()
    assert "ATL" in ratings
    assert "BOS" in ratings
    # Preseason row should be excluded by default.
    assert engine.games_processed == 2
    assert out_path.exists()

    payload = json.loads(out_path.read_text())
    assert payload["games_processed"] == 2
    assert "ATL" in payload["ratings"]
    assert "BOS" in payload["ratings"]

    p_home = engine.get_win_probability("ATL", "BOS", team_a_is_home=True)
    p_away = engine.get_win_probability("ATL", "BOS", team_a_is_home=False)
    assert 0.0 <= p_home <= 1.0
    assert 0.0 <= p_away <= 1.0
    assert p_home != p_away


def test_kalshi_matchup_parser_and_yes_probability():
    engine = EloEngine()
    engine.ratings = {
        "BKN": 1450.0,
        "CLE": 1650.0,
    }
    yes_prob = engine.get_market_yes_probability("KXNBAGAME-26FEB19BKNCLE-BKN")
    assert yes_prob is not None
    # YES is BKN (away underdog), so probability should be below 50%.
    assert yes_prob < 0.5

    cle_yes = engine.get_market_yes_probability("KXNBAGAME-26FEB19BKNCLE-CLE")
    assert cle_yes is not None
    # Complementary contracts on same event.
    assert abs((yes_prob + cle_yes) - 1.0) < 1e-6


def test_parse_elo_adjusted_estimate_clamps_delta():
    text = json.dumps(
        {
            "delta": 0.25,  # Should clamp to max_delta.
            "confidence": 85,
            "reasoning": "Major injuries favor the underdog.",
            "key_factors": ["injury report"],
            "data_sources": ["official report"],
        }
    )
    est = ClaudeResponseParser.parse_elo_adjusted_estimate(
        response_text=text,
        market_id="KXNBAGAME-26FEB19BKNCLE-BKN",
        market_price=0.40,
        base_probability=0.45,
        max_delta=0.03,
    )
    assert est is not None
    assert abs(est.estimated_probability - 0.48) < 1e-6
    assert abs(est.edge - 0.08) < 1e-6
    assert est.confidence_level == 0.85
