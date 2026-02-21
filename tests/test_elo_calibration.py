"""Tests for empirical Elo calibration helpers."""

import pandas as pd
import pytest

from app.analytics.elo_calibration import (
    EloCalibrationConfig,
    blend_probabilities,
    build_calibration_table,
    lookup_empirical_rate,
)


def test_blend_probabilities_weighting() -> None:
    p_final, w = blend_probabilities(p_elo=0.62, p_emp=0.90, n=0, prior=100)
    assert w == 0.0
    assert p_final == pytest.approx(0.62)

    p_final_mid, w_mid = blend_probabilities(p_elo=0.40, p_emp=0.80, n=100, prior=100)
    assert w_mid == pytest.approx(0.5)
    assert p_final_mid == pytest.approx(0.60)


def test_bucket_lookup_and_fallback() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "season": [2024, 2024, 2024],
            "is_home": [1, 1, 0],
            "elo_difference": [12.0, 20.0, -30.0],
            "result": [1, 0, 1],
        }
    )
    cfg = EloCalibrationConfig(bucket_size=25, prior=100)
    table = build_calibration_table(df, cfg)

    p_emp, n, bucket_key = lookup_empirical_rate(table, is_home=1, elo_difference=24.9, bucket_size=25)
    assert bucket_key == "home=1|diff=0"
    assert p_emp == pytest.approx(0.5)
    assert n == pytest.approx(2.0)

    p_emp_missing, n_missing, missing_key = lookup_empirical_rate(
        table,
        is_home=1,
        elo_difference=200.0,
        bucket_size=25,
    )
    assert missing_key == "home=1|diff=200"
    assert p_emp_missing is None
    assert n_missing == 0.0


def test_cutoff_date_filters_out_future_rows() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-10"],
            "season": [2024, 2024],
            "is_home": [1, 1],
            "elo_difference": [12.0, 15.0],
            "result": [1, 0],
        }
    )
    cfg = EloCalibrationConfig(bucket_size=25, prior=100)
    full_table = build_calibration_table(df, cfg)
    cutoff_table = build_calibration_table(df, cfg, cutoff_date="2024-01-05")

    _, full_n, _ = lookup_empirical_rate(full_table, is_home=1, elo_difference=20.0, bucket_size=25)
    _, cutoff_n, _ = lookup_empirical_rate(cutoff_table, is_home=1, elo_difference=20.0, bucket_size=25)

    assert full_n == pytest.approx(2.0)
    assert cutoff_n == pytest.approx(1.0)


def test_recency_weighting_changes_p_emp() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-11"],
            "season": [2024, 2024],
            "is_home": [1, 1],
            "elo_difference": [10.0, 8.0],
            "result": [0, 1],
        }
    )
    cfg = EloCalibrationConfig(
        bucket_size=25,
        prior=100,
        recency_mode="exp",
        recency_halflife_days=1,
    )
    weighted_table = build_calibration_table(df, cfg, cutoff_date="2024-01-12")
    row = weighted_table.loc[(1, 0)]

    assert float(row["raw_n"]) == pytest.approx(2.0)
    assert float(row["p_emp"]) > 0.9
