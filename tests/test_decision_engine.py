from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.models.fair_value_estimate import FairValueEstimate
from app.models.market_data import MarketData
from app.trading.decision_engine import (
    append_decision_jsonl,
    choose_side_from_estimate,
    compute_net_edge,
    compute_pre_fee_edge,
    evaluate_opportunity,
)


def _config(
    *,
    min_edge_net: float = 0.025,
    min_confidence: float = 0.75,
    taker_fee_pct: float = 0.0075,
    slippage_pct: float = 0.005,
):
    return SimpleNamespace(
        strategy=SimpleNamespace(
            min_edge_net=min_edge_net,
            min_confidence=min_confidence,
        ),
        execution=SimpleNamespace(
            kalshi_taker_fee_pct=taker_fee_pct,
            estimated_slippage_pct=slippage_pct,
            use_dynamic_slippage=False,
        ),
    )


def _market(*, yes_price: float, no_price: float) -> MarketData:
    return MarketData(
        platform="kalshi",
        market_id="KXNBAGAME-26FEB23SACMEM-MEM",
        title="Will Memphis beat Sacramento?",
        description="NBA game winner market",
        yes_price=yes_price,
        no_price=no_price,
        volume=10000.0,
        liquidity=2000.0,
        end_date="2026-02-23T23:59:59Z",
        category="sports",
        event_ticker="KXNBAGAME-26FEB23SACMEM",
        series_ticker="KXNBAGAME",
        yes_option="Memphis",
        no_option="Sacramento",
    )


def _estimate(*, p: float, conf: float, edge: float = 0.0) -> FairValueEstimate:
    return FairValueEstimate(
        market_id="KXNBAGAME-26FEB23SACMEM-MEM",
        estimated_probability=p,
        confidence_level=conf,
        edge=edge,
        reasoning="test",
    )


def test_compute_pre_fee_edge_yes_no():
    yes_edge = compute_pre_fee_edge("yes", fair_prob=0.61, yes_price=0.55, no_price=0.45)
    no_edge = compute_pre_fee_edge("no", fair_prob=0.61, yes_price=0.55, no_price=0.45)
    assert yes_edge == pytest.approx(0.06)
    assert no_edge == pytest.approx(-0.06)


def test_compute_net_edge_subtracts_costs():
    assert compute_net_edge(0.08, 0.0125) == 0.0675


def test_choose_side_prefers_higher_edge():
    assert choose_side_from_estimate(fair_prob=0.70, yes_price=0.55, no_price=0.45) == "yes"
    assert choose_side_from_estimate(fair_prob=0.30, yes_price=0.55, no_price=0.45) == "no"


def test_evaluate_opportunity_skip_low_edge():
    market = _market(yes_price=0.55, no_price=0.45)
    estimate = _estimate(p=0.565, conf=0.90, edge=0.015)
    record = evaluate_opportunity(market, estimate, _config(min_edge_net=0.03))
    assert record.decision == "skip"
    assert record.reason == "skip_low_net_edge"


def test_evaluate_opportunity_skip_low_confidence():
    market = _market(yes_price=0.50, no_price=0.50)
    estimate = _estimate(p=0.65, conf=0.60, edge=0.15)
    record = evaluate_opportunity(market, estimate, _config(min_confidence=0.75))
    assert record.decision == "skip"
    assert record.reason == "skip_low_confidence"


def test_evaluate_opportunity_take_yes_or_no():
    cfg = _config(min_edge_net=0.02, min_confidence=0.70, taker_fee_pct=0.002, slippage_pct=0.001)

    market_yes = _market(yes_price=0.45, no_price=0.55)
    estimate_yes = _estimate(p=0.60, conf=0.85, edge=0.15)
    record_yes = evaluate_opportunity(market_yes, estimate_yes, cfg)
    assert record_yes.decision == "take"
    assert record_yes.reason == "take_strong_yes"
    assert record_yes.side_considered == "yes"

    market_no = _market(yes_price=0.65, no_price=0.35)
    estimate_no = _estimate(p=0.40, conf=0.85, edge=-0.25)
    record_no = evaluate_opportunity(market_no, estimate_no, cfg)
    assert record_no.decision == "take"
    assert record_no.reason == "take_strong_no"
    assert record_no.side_considered == "no"


def test_append_decision_jsonl_roundtrip(tmp_path):
    market = _market(yes_price=0.45, no_price=0.55)
    estimate = _estimate(p=0.60, conf=0.85, edge=0.15)
    record = evaluate_opportunity(market, estimate, _config())

    out_path = tmp_path / "opportunities.jsonl"
    append_decision_jsonl(record, str(out_path))

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["market_id"] == market.market_id
    assert payload["decision"] in {"take", "skip"}
    assert "attribution" in payload
