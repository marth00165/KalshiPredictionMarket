# Kalshi NBA Bot Go-Live Plan

## Objective
Move from dry-run research to controlled live deployment with measurable risk limits, execution guardrails, and clear rollback criteria.

## Scope
- Product: Kalshi NBA pregame winner markets (`KXNBAGAME`)
- Model stack: Elo + deterministic injury PIM + empirical calibration
- Excluded for initial launch: in-game trading, cross-market arbitrage, non-NBA series

## Success Criteria
- Positive net PnL after fees/slippage in shadow + paper windows
- Stable calibration (Brier/log loss no material degradation vs baseline)
- No risk limit breaches and no operational incidents in consecutive evaluation windows

## Phase 0: Configuration Lock
- Freeze core model parameters for test window:
  - `nba_elo_regression_factor=0.5`
  - calibration settings fixed for the window
  - injury feed source and parsing version fixed
- Snapshot config + commit hash before testing
- Regenerate Elo artifacts whenever Elo params or source CSV changes:
  - `context/historical_elo_matchups.csv`
  - `app/outputs/elo_ratings_by_season.csv`
  - `app/outputs/elo_ratings.json`

## Phase 1: Shadow Mode (No Orders)
Duration: 2-4 weeks minimum.

Track per cycle:
- candidate count, traded count (simulated), skipped reasons
- average and distribution of model edge
- calibration stats: Brier, log loss, reliability bins
- realized win rate by edge bucket
- simulated PnL after fees/slippage assumptions

Exit gate:
- Positive simulated net PnL in each week OR positive aggregate with no single catastrophic week
- Stable calibration vs baseline (no persistent drift)

## Phase 2: Paper Execution Fidelity
Duration: 1-2 weeks.

Focus:
- Limit-order behavior assumptions
- Fill probability realism
- Cancel/replace logic under moving prices

Exit gate:
- Paper execution metrics align with expected fill model
- No unresolved execution bugs in logs

## Phase 3: Small-Capital Live Pilot
Initial constraints:
- Very small bankroll slice (e.g., 5-10% of allocated capital)
- Max positions low
- Strict max notional per cycle
- Higher edge threshold than dry run

Required controls:
- Daily loss stop
- Exposure cap
- Per-market max loss
- Immediate kill switch path tested

Exit gate:
- Positive net PnL after costs over minimum sample size
- No risk policy violations

## Phase 4: Gradual Scale-Up
- Increase limits in fixed steps (e.g., +25%) only after each stable window
- Keep guardrails constant until sufficient live sample
- Revert one step on underperformance or process failures

## Operational Guardrails
- Trade only NBA markets with adequate liquidity/spread
- No trading when injury feed freshness or model metadata is stale
- Reject trades if market moved beyond execution drift threshold
- Enforce duplicate/correlation controls

## Monitoring Dashboard (Minimum)
- PnL: gross, fees, net
- Fill rate and slippage vs expected
- Edge capture: expected edge vs realized outcome by bucket
- Model health: Brier/log loss, calibration tail gaps
- Risk: exposure %, drawdown, daily loss utilization

## Incident / Rollback Plan
Immediate rollback triggers:
- Daily loss limit hit
- Data feed integrity issue
- Repeated execution failures
- Model output anomalies (NaNs, invalid probabilities, metadata missing)

Rollback actions:
1. Set trading disabled via kill switch
2. Stop order submission loop
3. Preserve reports/logs for postmortem
4. Revert to last known-good commit/config

## Weekly Review Checklist
- Compare live/paper performance vs shadow expectation
- Recompute calibration and drift diagnostics
- Review top winners/losers and failure causes
- Decide: keep, tighten, or roll back limits

## Immediate Next Actions
1. Keep `regression_factor=0.5` for the next evaluation block
2. Run daily dry-run analysis and archive reports
3. Add a go-live readiness script that validates:
   - artifact freshness
   - config hash
   - key risk thresholds present
