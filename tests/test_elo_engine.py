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


def _write_two_season_games_csv(path):
    path.write_text(
        "\n".join(
            [
                "gameId,gameDateTimeEst,hometeamId,awayteamId,homeScore,awayScore,gameType",
                "1,2024-04-01 19:00:00,1610612737,1610612738,110,100,Regular Season",
                "2,2024-10-01 19:00:00,1610612738,1610612737,100,90,Regular Season",
            ]
        )
        + "\n"
    )


def _write_season_cutoff_games_csv(path):
    path.write_text(
        "\n".join(
            [
                "gameId,gameDateTimeEst,hometeamId,awayteamId,homeScore,awayScore,gameType",
                "1,2024-09-30 19:00:00,1610612737,1610612738,100,90,Regular Season",
                "2,2024-10-01 19:00:00,1610612737,1610612738,101,90,Regular Season",
            ]
        )
        + "\n"
    )


def test_elo_engine_rebuild_exports_and_probabilities(tmp_path):
    data_path = tmp_path / "kaggleGameData.csv"
    out_path = tmp_path / "elo_ratings.json"
    season_out_path = tmp_path / "elo_ratings_by_season.csv"
    _write_sample_games_csv(data_path)

    engine = EloEngine(
        data_path=str(data_path),
        output_path=str(out_path),
        season_ratings_output_path=str(season_out_path),
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
    assert season_out_path.exists()

    payload = json.loads(out_path.read_text())
    assert payload["games_processed"] == 2
    assert "ATL" in payload["ratings"]
    assert "BOS" in payload["ratings"]
    season_rows = season_out_path.read_text().strip().splitlines()
    assert season_rows[0] == "season,team,elo_final"
    assert any(row.startswith("2024,ATL,") for row in season_rows[1:])
    assert any(row.startswith("2024,BOS,") for row in season_rows[1:])

    p_home = engine.get_win_probability("ATL", "BOS", team_a_is_home=True)
    p_away = engine.get_win_probability("ATL", "BOS", team_a_is_home=False)
    assert 0.0 <= p_home <= 1.0
    assert 0.0 <= p_away <= 1.0
    assert p_home != p_away


def test_elo_engine_applies_offseason_regression(tmp_path):
    data_path = tmp_path / "games_two_seasons.csv"
    out_path = tmp_path / "elo_ratings.json"
    season_out_path = tmp_path / "elo_ratings_by_season.csv"
    _write_two_season_games_csv(data_path)

    engine = EloEngine(
        data_path=str(data_path),
        output_path=str(out_path),
        season_ratings_output_path=str(season_out_path),
        initial_rating=1500.0,
        home_advantage=0.0,
        k_factor=20.0,
        regression_factor=0.5,
        use_mov_multiplier=False,
        elo_round_decimals=3,
        min_season=None,
    )
    ratings = engine.rebuild()

    # After game1: ATL=1510, BOS=1490
    # Season rollover regression r=0.5 -> ATL=1505, BOS=1495
    # Game2 expected BOS = 1/(1+10^((1505-1495)/400)) ~= 0.4856128
    # BOS new ~= 1505.288, ATL new ~= 1494.712
    assert ratings["BOS"] == 1505.288
    assert ratings["ATL"] == 1494.712


def test_elo_engine_respects_min_season_filter(tmp_path):
    data_path = tmp_path / "games_cutoff.csv"
    out_path = tmp_path / "elo_ratings.json"
    season_out_path = tmp_path / "elo_ratings_by_season.csv"
    _write_season_cutoff_games_csv(data_path)

    engine = EloEngine(
        data_path=str(data_path),
        output_path=str(out_path),
        season_ratings_output_path=str(season_out_path),
        initial_rating=1500.0,
        home_advantage=0.0,
        k_factor=20.0,
        regression_factor=0.75,
        use_mov_multiplier=False,
        elo_round_decimals=3,
        min_season=2025,
    )
    engine.rebuild()

    # 2024-09-30 belongs to season 2024; 2024-10-01 belongs to season 2025.
    assert engine.games_processed == 1


def test_elo_engine_respects_allowed_seasons_filter(tmp_path):
    data_path = tmp_path / "games_cutoff.csv"
    out_path = tmp_path / "elo_ratings.json"
    season_out_path = tmp_path / "elo_ratings_by_season.csv"
    _write_season_cutoff_games_csv(data_path)

    engine = EloEngine(
        data_path=str(data_path),
        output_path=str(out_path),
        season_ratings_output_path=str(season_out_path),
        initial_rating=1500.0,
        home_advantage=0.0,
        k_factor=20.0,
        regression_factor=0.75,
        use_mov_multiplier=False,
        elo_round_decimals=3,
        min_season=None,
        allowed_seasons=[2025],
    )
    engine.rebuild()

    assert engine.games_processed == 1


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
            "elo_delta": 300,  # Should clamp to +75.
            "confidence": 0.85,
            "reason": "Major injuries favor the underdog.",
            "key_factors": ["injury report"],
            "data_sources": ["official report"],
        }
    )
    est = ClaudeResponseParser.parse_elo_adjusted_estimate(
        response_text=text,
        market_id="KXNBAGAME-26FEB19BKNCLE-BKN",
        market_price=0.40,
        yes_team="BKN",
        home_team="CLE",
        away_team="BKN",
        home_elo=1650.0,
        away_elo=1450.0,
        home_court_bonus=100.0,
    )
    assert est is not None
    # Away team gets +75 Elo, still underdog versus home+HCA.
    assert 0.0 < est.estimated_probability < 0.5
    assert abs(est.edge - (est.estimated_probability - 0.40)) < 1e-9
    assert est.confidence_level == 0.85
    meta = (est.fusion_metadata or {}).get("elo_adjustment", {})
    suggestion = meta.get("llm_suggestion", {})
    assert suggestion.get("applied_elo_delta") == 75.0
    assert suggestion.get("raw_elo_delta") == 300
    assert suggestion.get("injury_report", {}).get("status") == "unknown"


def test_parse_elo_adjusted_estimate_captures_injury_report_fields():
    text = json.dumps(
        {
            "elo_delta": -20,
            "confidence": 0.77,
            "reason": "Starting guard out, downgrade YES side.",
            "injury_report": {
                "status": "confirmed",
                "impact": "favors_no",
                "notes": "Starting guard ruled out pre-game.",
            },
            "key_factors": ["injury report", "lineup downgrade"],
            "data_sources": ["team injury report"],
        }
    )
    est = ClaudeResponseParser.parse_elo_adjusted_estimate(
        response_text=text,
        market_id="KXNBAGAME-26FEB19BKNCLE-BKN",
        market_price=0.40,
        yes_team="BKN",
        home_team="CLE",
        away_team="BKN",
        home_elo=1650.0,
        away_elo=1450.0,
        home_court_bonus=100.0,
    )
    assert est is not None
    suggestion = ((est.fusion_metadata or {}).get("elo_adjustment") or {}).get("llm_suggestion") or {}
    assert suggestion.get("applied_elo_delta") == -20.0
    assert suggestion.get("injury_report", {}).get("status") == "confirmed"
    assert suggestion.get("injury_report", {}).get("impact") == "favors_no"


def test_parse_elo_adjusted_estimate_missing_required_field_returns_none():
    text = json.dumps(
        {
            "elo_delta": -40,
            "confidence": 0.80,
            # reason intentionally omitted
        }
    )
    est = ClaudeResponseParser.parse_elo_adjusted_estimate(
        response_text=text,
        market_id="KXNBAGAME-26FEB19BKNCLE-BKN",
        market_price=0.40,
        yes_team="BKN",
        home_team="CLE",
        away_team="BKN",
        home_elo=1650.0,
        away_elo=1450.0,
        home_court_bonus=100.0,
    )
    assert est is None


def test_elo_ratings_payload_param_guard(tmp_path):
    out_path = tmp_path / "elo_ratings.json"
    season_out_path = tmp_path / "elo_ratings_by_season.csv"
    season_out_path.write_text("season,team,elo_final\n")

    mismatched_payload = {
        "model_version": EloEngine.MODEL_VERSION,
        "last_updated": "2000-01-01T00:00:00Z",
        "params": {
            "initial_rating": 1500.0,
            "home_advantage": 100.0,
            "k_factor": 21.0,
            "regression_factor": 0.75,
            "use_mov_multiplier": True,
            "elo_round_decimals": 1,
            "min_season": 2004,
            "allowed_seasons": None,
        },
        "ratings": {"DEN": 1700.0},
        "games_processed": 1,
    }
    out_path.write_text(json.dumps(mismatched_payload))

    engine_mismatched = EloEngine(
        data_path=str(tmp_path / "missing.csv"),
        output_path=str(out_path),
        season_ratings_output_path=str(season_out_path),
        max_age_hours=24,
    )
    assert engine_mismatched.load_ratings(rebuild_if_missing=False) == {}

    matching_payload = {
        "model_version": EloEngine.MODEL_VERSION,
        "last_updated": "2000-01-01T00:00:00Z",
        "params": {
            "initial_rating": 1500.0,
            "home_advantage": 100.0,
            "k_factor": 20.0,
            "regression_factor": 0.75,
            "use_mov_multiplier": True,
            "elo_round_decimals": 1,
            "min_season": 2004,
            "allowed_seasons": None,
        },
        "ratings": {"DEN": 1700.0},
        "games_processed": 1,
    }
    out_path.write_text(json.dumps(matching_payload))

    engine_matching = EloEngine(
        data_path=str(tmp_path / "missing.csv"),
        output_path=str(out_path),
        season_ratings_output_path=str(season_out_path),
        max_age_hours=24,
    )
    loaded = engine_matching.load_ratings(rebuild_if_missing=False)
    assert loaded["DEN"] == 1700.0
