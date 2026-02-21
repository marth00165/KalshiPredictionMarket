"""Empirical calibration helpers for NBA Elo probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import pandas as pd


@dataclass(frozen=True)
class EloCalibrationConfig:
    """Configuration for empirical Elo probability calibration."""

    bucket_size: int = 25
    prior: int = 100
    key_type: str = "elo_diff"
    min_season: Optional[int] = None
    recency_mode: str = "none"  # "none" | "exp"
    recency_halflife_days: int = 365

    def __post_init__(self) -> None:
        if self.bucket_size <= 0:
            raise ValueError("bucket_size must be > 0")
        if self.prior < 0:
            raise ValueError("prior must be >= 0")
        if self.key_type != "elo_diff":
            raise ValueError("Only key_type='elo_diff' is currently supported")
        mode = str(self.recency_mode or "none").strip().lower()
        if mode not in {"none", "exp"}:
            raise ValueError("recency_mode must be one of: none, exp")
        if self.recency_halflife_days <= 0:
            raise ValueError("recency_halflife_days must be > 0")


def load_matchups_csv(path: str) -> pd.DataFrame:
    """Load historical matchup rows from CSV and coerce core dtypes."""
    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError("Calibration CSV must include a 'date' column")
    if "is_home" not in df.columns:
        raise ValueError("Calibration CSV must include an 'is_home' column")
    if "result" not in df.columns:
        raise ValueError("Calibration CSV must include a 'result' column")
    if "elo_difference" not in df.columns:
        raise ValueError("Calibration CSV must include an 'elo_difference' column")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["is_home"] = pd.to_numeric(df["is_home"], errors="coerce").round().clip(0, 1).astype("Int64")
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df["elo_difference"] = pd.to_numeric(df["elo_difference"], errors="coerce")

    if "season" in df.columns:
        season_numeric = pd.to_numeric(df["season"], errors="coerce")
        df["season"] = season_numeric.astype("Int64")

    return df


def build_calibration_table(
    df: pd.DataFrame,
    config: EloCalibrationConfig,
    cutoff_date: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    Build empirical win-rate table keyed by (is_home, elo_diff_bucket).

    Returns a DataFrame indexed by (is_home, elo_diff_bucket) with:
    - n: sample size (effective sample size when recency weighting is enabled)
    - p_emp: empirical win rate
    - raw_n: optional raw count (only present for recency_mode='exp')
    """
    if df is None or df.empty:
        return _empty_table(include_raw_n=(config.recency_mode == "exp"))

    working = df.copy()
    working["date"] = pd.to_datetime(working.get("date"), errors="coerce")
    working["is_home"] = pd.to_numeric(working.get("is_home"), errors="coerce")
    working["result"] = pd.to_numeric(working.get("result"), errors="coerce")
    working["elo_difference"] = pd.to_numeric(working.get("elo_difference"), errors="coerce")

    working = working.dropna(subset=["date", "is_home", "result", "elo_difference"])
    if working.empty:
        return _empty_table(include_raw_n=(config.recency_mode == "exp"))

    working["is_home"] = working["is_home"].astype(int).clip(0, 1)
    working["elo_diff_bucket"] = _bucket_series(working["elo_difference"], config.bucket_size)

    cutoff_ts: Optional[pd.Timestamp] = None
    if cutoff_date is not None:
        cutoff_ts = pd.to_datetime(cutoff_date, errors="coerce")
        if pd.isna(cutoff_ts):
            raise ValueError("Invalid cutoff_date provided")
        working = working[working["date"] < cutoff_ts]

    if config.min_season is not None and "season" in working.columns:
        seasons = pd.to_numeric(working["season"], errors="coerce")
        working = working.loc[seasons >= int(config.min_season)]

    if working.empty:
        return _empty_table(include_raw_n=(config.recency_mode == "exp"))

    mode = str(config.recency_mode or "none").strip().lower()
    if mode == "exp":
        ref_date = (cutoff_ts - pd.Timedelta(days=1)) if cutoff_ts is not None else working["date"].max()
        age_days = (ref_date - working["date"]).dt.days.clip(lower=0)
        weights = 0.5 ** (age_days / float(config.recency_halflife_days))

        weighted = working.assign(
            calibration_weight=weights,
            weighted_result=weights * working["result"],
        )

        grouped = (
            weighted.groupby(["is_home", "elo_diff_bucket"], as_index=False)
            .agg(
                n=("calibration_weight", "sum"),
                weighted_result_sum=("weighted_result", "sum"),
                raw_n=("result", "count"),
            )
        )
        grouped["p_emp"] = grouped["weighted_result_sum"] / grouped["n"]
        grouped = grouped.drop(columns=["weighted_result_sum"])
    else:
        grouped = (
            working.groupby(["is_home", "elo_diff_bucket"], as_index=False)
            .agg(
                n=("result", "count"),
                p_emp=("result", "mean"),
            )
        )

    grouped = grouped.sort_values(["is_home", "elo_diff_bucket"]).reset_index(drop=True)
    grouped["n"] = pd.to_numeric(grouped["n"], errors="coerce").fillna(0.0)
    grouped["p_emp"] = pd.to_numeric(grouped["p_emp"], errors="coerce")
    grouped = grouped.set_index(["is_home", "elo_diff_bucket"])
    return grouped


def lookup_empirical_rate(
    cal_table: Optional[pd.DataFrame],
    is_home: int,
    elo_difference: float,
    bucket_size: int = 25,
) -> Tuple[Optional[float], float, str]:
    """Lookup empirical rate by home/away flag and Elo-difference bucket."""
    home_flag = int(is_home)
    bucket = _bucket_value(float(elo_difference), bucket_size)
    bucket_key = f"home={home_flag}|diff={bucket}"

    if cal_table is None or cal_table.empty:
        return None, 0.0, bucket_key

    try:
        row = cal_table.loc[(home_flag, bucket)]
    except KeyError:
        return None, 0.0, bucket_key

    if isinstance(row, pd.DataFrame):
        if row.empty:
            return None, 0.0, bucket_key
        row_data = row.iloc[0]
    else:
        row_data = row

    p_emp_raw = row_data.get("p_emp")
    n_raw = row_data.get("n", 0.0)

    if p_emp_raw is None or pd.isna(p_emp_raw):
        return None, 0.0, bucket_key

    try:
        return float(p_emp_raw), float(n_raw), bucket_key
    except (TypeError, ValueError):
        return None, 0.0, bucket_key


def blend_probabilities(
    p_elo: float,
    p_emp: Optional[float],
    n: float,
    prior: float,
) -> Tuple[float, float]:
    """Blend model probability with empirical bucket win-rate using shrinkage."""
    p_elo_clamped = _clamp_probability(p_elo)

    try:
        n_value = max(float(n), 0.0)
    except (TypeError, ValueError):
        n_value = 0.0

    try:
        prior_value = max(float(prior), 0.0)
    except (TypeError, ValueError):
        prior_value = 0.0

    if p_emp is None or n_value <= 0.0:
        return p_elo_clamped, 0.0

    p_emp_clamped = _clamp_probability(float(p_emp))
    denom = n_value + prior_value
    w = (n_value / denom) if denom > 0 else 1.0
    p_final = (w * p_emp_clamped) + ((1.0 - w) * p_elo_clamped)
    return _clamp_probability(p_final), float(max(0.0, min(1.0, w)))


def _bucket_series(values: pd.Series, bucket_size: int) -> pd.Series:
    return (values.floordiv(float(bucket_size)) * float(bucket_size)).astype(int)


def _bucket_value(value: float, bucket_size: int) -> int:
    return int((value // float(bucket_size)) * float(bucket_size))


def _clamp_probability(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _empty_table(include_raw_n: bool = False) -> pd.DataFrame:
    columns = ["n", "p_emp"]
    if include_raw_n:
        columns.append("raw_n")
    empty = pd.DataFrame(columns=columns)
    empty.index = pd.MultiIndex.from_arrays([[], []], names=["is_home", "elo_diff_bucket"])
    return empty
