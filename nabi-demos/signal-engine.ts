#!/usr/bin/env bun
/**
 * Polymarket Signal Engine — ML Feature Extraction + Trade Signals
 *
 * Periodically queries SurrealDB for recent polymarket.live events,
 * computes time-windowed features per market, and generates
 * buy/sell/hold signals published back to the substrate.
 *
 * This is the "brain" that feeds into the virtual portfolio tracker.
 * The same pipeline that proved domain-agnostic (football → crypto → predictions)
 * now produces actionable trade signals from prediction market data.
 *
 * Usage:
 *   bun run signal-engine.ts                     # Default: 15m window, 60s cycle
 *   bun run signal-engine.ts --window 30         # 30-minute analysis window
 *   bun run signal-engine.ts --cycle 30          # Run every 30 seconds
 *   bun run signal-engine.ts --min-trades 5      # Require 5 trades minimum
 *   bun run signal-engine.ts --dry-run           # Print signals without publishing
 *   bun run signal-engine.ts --once              # Single pass, then exit
 *
 * Substrate Mapping:
 *   feature extraction → source: polymarket.signals
 *   trade suggestion   → event_type: signal.suggestion.{buy|sell|hold}
 *   portfolio action   → event_type: signal.portfolio.{entry|exit}
 */

// ─── Config ─────────────────────────────────────────────────────────────────

const args = process.argv.slice(2)
function getArg(name: string, fallback: string): string {
  const idx = args.indexOf(`--${name}`)
  return idx >= 0 && args[idx + 1] ? args[idx + 1] : fallback
}
const hasFlag = (name: string) => args.includes(`--${name}`)

const WINDOW_MINUTES = parseInt(getArg('window', '15'))
const CYCLE_SECONDS = parseInt(getArg('cycle', '60'))
const MIN_TRADES = parseInt(getArg('min-trades', '3'))
const DRY_RUN = hasFlag('dry-run')
const ONCE = hasFlag('once')

const SURREAL_URL = 'http://127.0.0.1:8284'
const SURREAL_NS = 'nabi'
const SURREAL_DB = 'substrate'
const SURREAL_AUTH = Buffer.from('root:root').toString('base64')

const SOURCE = 'polymarket.signals'
const PLEXUS_API = 'http://localhost:5381/events/full'

// ─── Types ──────────────────────────────────────────────────────────────────

interface TradeEvent {
  event_type: string
  severity: string
  message: string
  timestamp: string
  metadata: {
    condition_id: string
    market_slug: string
    outcome: string
    price: string
    probability_pct: string
    size: string
    side: string
    seq: number
    derived?: {
      price_delta?: string
      severity_reason?: string
    }
  }
}

interface MarketFeatures {
  condition_id: string
  outcome: string
  market_slug: string
  trade_count: number
  // Price features
  current_prob: number
  start_prob: number
  price_momentum: number       // Linear trend slope (positive = bullish)
  price_volatility: number     // Std dev of probability changes
  price_acceleration: number   // Change in momentum (second derivative)
  // Volume features
  total_volume: number
  buy_volume: number
  sell_volume: number
  volume_imbalance: number     // (buy - sell) / total, range [-1, 1]
  avg_trade_size: number
  large_trade_ratio: number    // Fraction from trades > 2σ
  // Activity features
  trades_per_minute: number
  last_trade_age_sec: number
  // Severity features
  warning_count: number
  critical_count: number
}

type Signal = 'strong_buy' | 'buy' | 'hold' | 'sell' | 'strong_sell'

interface TradeSignal {
  condition_id: string
  outcome: string
  market_slug: string
  signal: Signal
  confidence: number           // 0-1
  features: MarketFeatures
  reasoning: string[]
}

// ─── SurrealDB Query ────────────────────────────────────────────────────────

async function queryTrades(windowMinutes: number): Promise<TradeEvent[]> {
  const cutoff = new Date(Date.now() - windowMinutes * 60 * 1000).toISOString()
  const query = `SELECT event_type, severity, message, timestamp, metadata FROM plexus_event WHERE source = 'polymarket.live' AND timestamp > '${cutoff}' ORDER BY timestamp ASC`

  const response = await fetch(`${SURREAL_URL}/sql`, {
    method: 'POST',
    headers: {
      'Authorization': `Basic ${SURREAL_AUTH}`,
      'surreal-ns': SURREAL_NS,
      'surreal-db': SURREAL_DB,
      'Accept': 'application/json',
    },
    body: query,
  })

  if (!response.ok) {
    throw new Error(`SurrealDB query failed: ${response.status}`)
  }

  const results: any[] = await response.json()
  if (results[0]?.status !== 'OK') {
    throw new Error(`SurrealDB query error: ${JSON.stringify(results[0])}`)
  }

  return results[0].result || []
}

// ─── Feature Extraction ─────────────────────────────────────────────────────

function extractFeatures(trades: TradeEvent[], windowMinutes: number): MarketFeatures[] {
  // Group trades by asset (conditionId:outcome)
  const grouped = new Map<string, TradeEvent[]>()

  for (const t of trades) {
    const key = `${t.metadata.condition_id}:${t.metadata.outcome}`
    if (!grouped.has(key)) grouped.set(key, [])
    grouped.get(key)!.push(t)
  }

  const features: MarketFeatures[] = []

  for (const [key, marketTrades] of grouped) {
    if (marketTrades.length < MIN_TRADES) continue

    const prices = marketTrades.map(t => parseFloat(t.metadata.price))
    const sizes = marketTrades.map(t => parseFloat(t.metadata.size))
    const sides = marketTrades.map(t => t.metadata.side?.toUpperCase())
    const timestamps = marketTrades.map(t => new Date(t.timestamp).getTime())

    const first = marketTrades[0].metadata
    const last = marketTrades[marketTrades.length - 1].metadata
    const currentProb = parseFloat(last.price)
    const startProb = parseFloat(first.price)

    // Price momentum: linear regression slope on probability over time
    const momentum = linearRegressionSlope(timestamps, prices)

    // Price volatility: std dev of consecutive price changes
    const priceChanges = prices.slice(1).map((p, i) => p - prices[i])
    const volatility = stdDev(priceChanges)

    // Price acceleration: difference in momentum between halves
    const mid = Math.floor(prices.length / 2)
    if (mid > 1) {
      var firstHalfMomentum = linearRegressionSlope(timestamps.slice(0, mid), prices.slice(0, mid))
      var secondHalfMomentum = linearRegressionSlope(timestamps.slice(mid), prices.slice(mid))
    } else {
      var firstHalfMomentum = 0
      var secondHalfMomentum = momentum
    }
    const acceleration = secondHalfMomentum - firstHalfMomentum

    // Volume analysis
    const totalVolume = sizes.reduce((a, b) => a + b, 0)
    const buyVolume = sizes.filter((_, i) => sides[i] === 'BUY').reduce((a, b) => a + b, 0)
    const sellVolume = sizes.filter((_, i) => sides[i] === 'SELL').reduce((a, b) => a + b, 0)
    const volumeImbalance = totalVolume > 0 ? (buyVolume - sellVolume) / totalVolume : 0

    // Large trade detection
    const avgSize = totalVolume / sizes.length
    const sizeStdDev = stdDev(sizes)
    const largeThreshold = avgSize + 2 * sizeStdDev
    const largeTradeVolume = sizes.filter(s => s > largeThreshold).reduce((a, b) => a + b, 0)
    const largeTradeRatio = totalVolume > 0 ? largeTradeVolume / totalVolume : 0

    // Activity
    const spanMs = timestamps[timestamps.length - 1] - timestamps[0]
    const spanMinutes = Math.max(spanMs / 60000, 1)
    const tradesPerMinute = marketTrades.length / spanMinutes

    const lastTradeAge = (Date.now() - timestamps[timestamps.length - 1]) / 1000

    // Severity counts
    const warningCount = marketTrades.filter(t => t.severity === 'warning').length
    const criticalCount = marketTrades.filter(t => t.severity === 'critical').length

    features.push({
      condition_id: first.condition_id,
      outcome: first.outcome,
      market_slug: first.market_slug,
      trade_count: marketTrades.length,
      current_prob: currentProb,
      start_prob: startProb,
      price_momentum: momentum,
      price_volatility: volatility,
      price_acceleration: acceleration,
      total_volume: totalVolume,
      buy_volume: buyVolume,
      sell_volume: sellVolume,
      volume_imbalance: volumeImbalance,
      avg_trade_size: avgSize,
      large_trade_ratio: largeTradeRatio,
      trades_per_minute: tradesPerMinute,
      last_trade_age_sec: lastTradeAge,
      warning_count: warningCount,
      critical_count: criticalCount,
    })
  }

  // Sort by activity (most active markets first)
  features.sort((a, b) => b.trade_count - a.trade_count)

  return features
}

// ─── Signal Generation ──────────────────────────────────────────────────────

function generateSignals(features: MarketFeatures[]): TradeSignal[] {
  const signals: TradeSignal[] = []

  for (const f of features) {
    const reasoning: string[] = []
    let score = 0  // positive = bullish, negative = bearish

    // 1. Price Momentum (weight: 3x)
    //    In prediction markets, momentum IS the primary signal
    if (f.price_momentum > 0.001) {
      score += 3
      reasoning.push(`Positive momentum: +${(f.price_momentum * 100).toFixed(2)}%/min`)
    } else if (f.price_momentum < -0.001) {
      score -= 3
      reasoning.push(`Negative momentum: ${(f.price_momentum * 100).toFixed(2)}%/min`)
    }

    // 2. Volume Imbalance (weight: 2x)
    //    Strong buy pressure = smart money entering
    if (f.volume_imbalance > 0.3) {
      score += 2
      reasoning.push(`Buy pressure: ${(f.volume_imbalance * 100).toFixed(0)}% imbalance`)
    } else if (f.volume_imbalance < -0.3) {
      score -= 2
      reasoning.push(`Sell pressure: ${(f.volume_imbalance * 100).toFixed(0)}% imbalance`)
    }

    // 3. Price Acceleration (weight: 2x)
    //    Accelerating momentum = trend strengthening
    if (f.price_acceleration > 0 && f.price_momentum > 0) {
      score += 2
      reasoning.push('Momentum accelerating (trend strengthening)')
    } else if (f.price_acceleration < 0 && f.price_momentum < 0) {
      score -= 2
      reasoning.push('Momentum decelerating (trend weakening)')
    }

    // 4. Large Trade Ratio (weight: 1x)
    //    Big players moving = informed flow
    if (f.large_trade_ratio > 0.3) {
      score += Math.sign(f.volume_imbalance || f.price_momentum) * 1
      reasoning.push(`Large trades: ${(f.large_trade_ratio * 100).toFixed(0)}% of volume`)
    }

    // 5. Volatility penalty (weight: -1x)
    //    High volatility = uncertainty = lower confidence
    if (f.price_volatility > 0.05) {
      reasoning.push(`High volatility: ${(f.price_volatility * 100).toFixed(1)}% (reduces confidence)`)
    }

    // 6. Activity bonus/penalty
    if (f.last_trade_age_sec > 300) {
      score *= 0.5  // Stale market, halve the signal
      reasoning.push('Market stale (>5min since last trade)')
    }

    // Convert score to signal
    let signal: Signal
    if (score >= 5) signal = 'strong_buy'
    else if (score >= 2) signal = 'buy'
    else if (score <= -5) signal = 'strong_sell'
    else if (score <= -2) signal = 'sell'
    else signal = 'hold'

    // Confidence based on score magnitude and volatility
    const rawConfidence = Math.min(Math.abs(score) / 8, 1)
    const volatilityDiscount = Math.max(0, 1 - f.price_volatility * 5)
    const confidence = rawConfidence * volatilityDiscount

    if (signal !== 'hold' || f.trade_count >= 10) {
      signals.push({
        condition_id: f.condition_id,
        outcome: f.outcome,
        market_slug: f.market_slug,
        signal,
        confidence,
        features: f,
        reasoning,
      })
    }
  }

  // Sort by confidence descending
  signals.sort((a, b) => b.confidence - a.confidence)

  return signals
}

// ─── Event Publisher ─────────────────────────────────────────────────────────

async function publishSignal(signal: TradeSignal): Promise<boolean> {
  const severity = signal.signal.includes('strong') ? 'warning'
    : signal.signal === 'hold' ? 'info'
    : 'info'

  const payload = {
    source: SOURCE,
    event_type: `signal.suggestion.${signal.signal.replace('_', '')}`,
    severity,
    message: formatSignalMessage(signal),
    payload: {
      condition_id: signal.condition_id,
      outcome: signal.outcome,
      market_slug: signal.market_slug,
      signal: signal.signal,
      confidence: parseFloat(signal.confidence.toFixed(3)),
      current_prob: signal.features.current_prob,
      price_momentum: parseFloat(signal.features.price_momentum.toFixed(6)),
      volume_imbalance: parseFloat(signal.features.volume_imbalance.toFixed(3)),
      trade_count: signal.features.trade_count,
      total_volume: parseFloat(signal.features.total_volume.toFixed(2)),
      reasoning: signal.reasoning,
      window_minutes: WINDOW_MINUTES,
    },
  }

  if (DRY_RUN) {
    const arrow = signal.signal.includes('buy') ? '▲' : signal.signal.includes('sell') ? '▼' : '─'
    const conf = (signal.confidence * 100).toFixed(0)
    console.log(`  ${arrow} ${signal.signal.toUpperCase().padEnd(12)} ${conf}% | ${signal.outcome}@${(signal.features.current_prob * 100).toFixed(1)}% | ${signal.reasoning[0] || ''}`)
    return true
  }

  try {
    const response = await fetch(PLEXUS_API, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    return response.ok
  } catch {
    return false
  }
}

function formatSignalMessage(s: TradeSignal): string {
  const slug = s.market_slug.length > 30 ? s.market_slug.slice(0, 30) + '...' : s.market_slug
  return `${s.signal.toUpperCase()} ${s.outcome}@${(s.features.current_prob * 100).toFixed(1)}% conf:${(s.confidence * 100).toFixed(0)}% (${slug})`
}

// ─── Math Utilities ─────────────────────────────────────────────────────────

function linearRegressionSlope(xs: number[], ys: number[]): number {
  const n = xs.length
  if (n < 2) return 0

  // Normalize timestamps to minutes from start
  const x0 = xs[0]
  const xNorm = xs.map(x => (x - x0) / 60000)

  const sumX = xNorm.reduce((a, b) => a + b, 0)
  const sumY = ys.reduce((a, b) => a + b, 0)
  const sumXY = xNorm.reduce((acc, x, i) => acc + x * ys[i], 0)
  const sumXX = xNorm.reduce((acc, x) => acc + x * x, 0)

  const denom = n * sumXX - sumX * sumX
  if (Math.abs(denom) < 1e-10) return 0

  return (n * sumXY - sumX * sumY) / denom
}

function stdDev(values: number[]): number {
  if (values.length < 2) return 0
  const mean = values.reduce((a, b) => a + b, 0) / values.length
  const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length
  return Math.sqrt(variance)
}

// ─── Main Loop ──────────────────────────────────────────────────────────────

async function runCycle(): Promise<void> {
  const startTime = Date.now()

  console.log(`\n[CYCLE] ${new Date().toISOString()} | window: ${WINDOW_MINUTES}m`)

  // 1. Query recent trades
  const trades = await queryTrades(WINDOW_MINUTES)
  console.log(`  Trades in window: ${trades.length}`)

  if (trades.length === 0) {
    console.log('  No trades in window, skipping.')
    return
  }

  // 2. Extract features per market
  const features = extractFeatures(trades, WINDOW_MINUTES)
  console.log(`  Markets with ${MIN_TRADES}+ trades: ${features.length}`)

  if (features.length === 0) {
    console.log('  No markets with enough trades, skipping.')
    return
  }

  // 3. Generate signals
  const signals = generateSignals(features)
  const actionable = signals.filter(s => s.signal !== 'hold')
  console.log(`  Signals: ${signals.length} total, ${actionable.length} actionable`)

  // 4. Print top signals
  if (actionable.length > 0) {
    console.log(`\n  ── Top Signals (${WINDOW_MINUTES}m window) ──`)
    for (const s of actionable.slice(0, 10)) {
      const arrow = s.signal.includes('buy') ? '▲' : '▼'
      const conf = (s.confidence * 100).toFixed(0)
      const prob = (s.features.current_prob * 100).toFixed(1)
      const momentum = (s.features.price_momentum * 100).toFixed(2)
      const vol = s.features.volume_imbalance > 0 ? `+${(s.features.volume_imbalance * 100).toFixed(0)}` : (s.features.volume_imbalance * 100).toFixed(0)
      console.log(`  ${arrow} ${s.signal.toUpperCase().padEnd(12)} ${conf.padStart(3)}% | ${s.outcome.padEnd(6)}@${prob.padStart(5)}% | mom:${momentum.padStart(6)} vol:${vol.padStart(4)}% | ${s.features.trade_count} trades`)
    }
  }

  // 5. Publish signals to substrate
  let published = 0
  for (const s of signals) {
    if (await publishSignal(s)) published++
  }

  const elapsed = Date.now() - startTime
  console.log(`\n  Published: ${published}/${signals.length} | Elapsed: ${elapsed}ms`)
}

// ─── Entry ──────────────────────────────────────────────────────────────────

console.log(`[SIGNAL ENGINE] Polymarket Signal Analyzer`)
console.log(`[CONFIG] Window: ${WINDOW_MINUTES}m | Cycle: ${CYCLE_SECONDS}s | Min trades: ${MIN_TRADES}`)
console.log(`[CONFIG] SurrealDB: ${SURREAL_URL} | Source: ${SOURCE}`)
if (DRY_RUN) console.log('[CONFIG] DRY RUN — signals printed, not published')
if (ONCE) console.log('[CONFIG] Single pass mode')

// Initial run
await runCycle()

if (!ONCE) {
  // Continuous monitoring loop
  setInterval(async () => {
    try {
      await runCycle()
    } catch (err: any) {
      console.error(`[ERROR] Cycle failed: ${err.message}`)
    }
  }, CYCLE_SECONDS * 1000)

  // Keep process alive
  process.on('SIGINT', () => {
    console.log('\n[SHUTDOWN] Signal engine stopped.')
    process.exit(0)
  })
}
