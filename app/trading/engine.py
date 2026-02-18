"""Compatibility wrapper exposing Elo functions for trading integration."""

from typing import Dict, Optional

from app.analytics import EloEngine, get_win_probability as _get_win_probability, load_ratings as _load_ratings


def load_ratings(
    data_path: str = EloEngine.DEFAULT_DATA_PATH,
    output_path: str = EloEngine.DEFAULT_OUTPUT_PATH,
    force_rebuild: bool = False,
) -> Dict[str, float]:
    return _load_ratings(
        data_path=data_path,
        output_path=output_path,
        force_rebuild=force_rebuild,
    )


def get_win_probability(
    team_a: str,
    team_b: str,
    team_a_is_home: bool = True,
    home_advantage: Optional[float] = None,
) -> float:
    return _get_win_probability(
        team_a=team_a,
        team_b=team_b,
        team_a_is_home=team_a_is_home,
        home_advantage=home_advantage,
    )

