#!/usr/bin/env python3
"""
Calibration utility for regime thresholds.

Scans recent curated signals (event-queue.jsonl) and outputs distribution
stats for delta_rolling_std, delta_velocity, and anomaly_score so thresholds
can be tuned per market category.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _load_curated_events(path: Path, max_lines: int = 50000) -> List[Dict]:
    events = []
    if not path.exists():
        return events
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if not ev.get("event_type", "").endswith(".curated"):
                continue
            events.append(ev)
            if len(events) >= max_lines:
                break
    return events


def _percentiles(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((pct / 100.0) * (len(values) - 1)))
    return float(values[idx])


def main():
    queue_path = Path.home() / ".local/state/nabi/kernel/event-queue.jsonl"
    events = _load_curated_events(queue_path)

    buckets = defaultdict(lambda: {"delta_rolling_std": [], "delta_velocity": [], "anomaly_score": []})

    for ev in events:
        payload = ((ev.get("payload") or {}).get("enrichment") or {})
        meta = ev.get("payload") or {}
        category = (meta.get("category") or "unknown").lower()
        drs = meta.get("delta_rolling_std")
        dv = meta.get("delta_velocity")
        anomaly = payload.get("anomaly_score")
        if isinstance(drs, (int, float)):
            buckets[category]["delta_rolling_std"].append(float(drs))
        if isinstance(dv, (int, float)):
            buckets[category]["delta_velocity"].append(float(dv))
        if isinstance(anomaly, (int, float)):
            buckets[category]["anomaly_score"].append(float(anomaly))

    print("category, metric, p50, p75, p90, p95, count")
    for category, metrics in buckets.items():
        for metric, values in metrics.items():
            if not values:
                continue
            p50 = _percentiles(values, 50)
            p75 = _percentiles(values, 75)
            p90 = _percentiles(values, 90)
            p95 = _percentiles(values, 95)
            print(f"{category}, {metric}, {p50:.6f}, {p75:.6f}, {p90:.6f}, {p95:.6f}, {len(values)}")


if __name__ == "__main__":
    main()
