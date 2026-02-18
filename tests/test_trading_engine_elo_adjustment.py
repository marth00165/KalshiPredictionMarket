from app.trading.engine import (
    apply_elo_adjustment,
    calculate_adjusted_yes_probability,
    calculate_probability_from_elo,
    validate_llm_elo_delta,
)


def test_validate_llm_elo_delta_clamps_and_defaults():
    assert validate_llm_elo_delta(10) == 10
    assert validate_llm_elo_delta(999) == 75
    assert validate_llm_elo_delta(-999) == -75
    assert validate_llm_elo_delta("bad") == 0


def test_calculate_probability_from_elo_monotonic():
    p = calculate_probability_from_elo(1700, 1600)
    q = calculate_probability_from_elo(1600, 1700)
    assert p > 0.5
    assert q < 0.5


def test_den_vs_gsw_negative_delta_reduces_probability():
    # Requested test shape from spec:
    # team=DEN, opponent=GSW, elo_base=1700, elo_delta=-75
    base = calculate_adjusted_yes_probability(
        yes_team="DEN",
        home_team="DEN",
        away_team="GSW",
        home_elo=1700.0,
        away_elo=1600.0,
        llm_elo_delta=0,
        home_court_bonus=100.0,
    )
    adjusted = calculate_adjusted_yes_probability(
        yes_team="DEN",
        home_team="DEN",
        away_team="GSW",
        home_elo=1700.0,
        away_elo=1600.0,
        llm_elo_delta=-75,
        home_court_bonus=100.0,
    )
    assert adjusted["applied_elo_delta"] == -75
    assert adjusted["yes_adjusted_elo"] == apply_elo_adjustment("DEN", 1700.0, -75)
    assert adjusted["yes_probability"] < base["yes_probability"]
    assert 0.01 <= adjusted["yes_probability"] <= 0.99
