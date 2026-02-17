#!/usr/bin/env bun
/**
 * Virtual Portfolio Tracker — $1K Prediction Market Simulator
 *
 * Reads signal events from the substrate, executes virtual trades,
 * and tracks P&L to measure how well the signal engine performs.
 *
 * This is the governance feedback loop:
 *   Signals → Positions → Outcomes → Authority Gradient Adjustment
 *
 * The substrate doesn't just analyze — it LEARNS by tracking which
 * signals led to profitable outcomes and which didn't.
 *
 * Usage:
 *   bun run portfolio.ts                    # Start tracker (default $1K)
 *   bun run portfolio.ts --capital 5000     # Start with $5K
 *   bun run portfolio.ts --cycle 30         # Check every 30s
 *   bun run portfolio.ts --max-position 100 # Max $100 per position
 *   bun run portfolio.ts --min-confidence 0.3  # Only trade signals > 30%
 *   bun run portfolio.ts --status           # Print current portfolio and exit
 *   bun run portfolio.ts --dry-run          # Simulate without publishing events
 *
 * Position Sizing (Kelly-inspired):
 *   size = capital * confidence * 0.05  (max 5% of capital per trade)
 *   Capped by --max-position
 *
 * Exit Conditions:
 *   - Profit target: +20% on position → close
 *   - Stop loss: -15% on position → close
 *   - Time decay: position open > 2h with no momentum → close
 *   - Signal reversal: opposite signal on same market → close
 */

// ─── Config ─────────────────────────────────────────────────────────────────

const args = process.argv.slice(2)
function getArg(name: string, fallback: string): string {
  const idx = args.indexOf(`--${name}`)
  return idx >= 0 && args[idx + 1] ? args[idx + 1] : fallback
}
const hasFlag = (name: string) => args.includes(`--${name}`)

const INITIAL_CAPITAL = parseFloat(getArg('capital', '1000'))
const CYCLE_SECONDS = parseInt(getArg('cycle', '60'))
const MAX_POSITION_SIZE = parseFloat(getArg('max-position', '100'))
const MIN_CONFIDENCE = parseFloat(getArg('min-confidence', '0.20'))
const STATUS_ONLY = hasFlag('status')
const DRY_RUN = hasFlag('dry-run')

const PROFIT_TARGET = 0.20   // +20% take profit
const STOP_LOSS = -0.15      // -15% stop loss
const TIME_DECAY_HOURS = 2   // Close stale positions after 2h
const MAX_OPEN_POSITIONS = 10
const POSITION_RISK = 0.05   // Max 5% of capital per trade

const SURREAL_URL = 'http://127.0.0.1:8284'
const SURREAL_NS = 'nabi'
const SURREAL_DB = 'substrate'
const SURREAL_AUTH = Buffer.from('root:root').toString('base64')
const PLEXUS_API = 'http://localhost:5381/events/full'
const SOURCE = 'polymarket.portfolio'

// ─── Types ──────────────────────────────────────────────────────────────────

interface Position {
  id: string                   // Unique position ID
  condition_id: string
  outcome: string
  market_slug: string
  direction: 'long' | 'short'  // long = bought shares, short = sold shares
  entry_price: number          // Probability at entry (0-1)
  current_price: number        // Latest probability
  size_usd: number             // USD value at entry
  shares: number               // shares = size_usd / entry_price
  opened_at: string            // ISO timestamp
  signal_confidence: number    // Confidence of the signal that triggered entry
  unrealized_pnl: number       // Current P&L in USD
  unrealized_pnl_pct: number   // Current P&L percentage
}

interface PortfolioState {
  capital: number              // Available cash
  initial_capital: number
  total_value: number          // cash + positions
  total_pnl: number            // Total realized + unrealized P&L
  realized_pnl: number         // Closed position P&L
  positions: Position[]
  closed_count: number
  win_count: number
  loss_count: number
  created_at: string
  last_updated: string
}

interface SignalEvent {
  event_type: string
  timestamp: string
  metadata: {
    condition_id: string
    outcome: string
    market_slug: string
    signal: string
    confidence: number
    current_prob: number
    price_momentum: number
    volume_imbalance: number
    trade_count: number
  }
}

// ─── Portfolio State (in-memory, persisted to SurrealDB) ────────────────────

const STATE_FILE = '/home/tryk/.local/state/nabi/portfolio/polymarket.json'

let portfolio: PortfolioState = {
  capital: INITIAL_CAPITAL,
  initial_capital: INITIAL_CAPITAL,
  total_value: INITIAL_CAPITAL,
  total_pnl: 0,
  realized_pnl: 0,
  positions: [],
  closed_count: 0,
  win_count: 0,
  loss_count: 0,
  created_at: new Date().toISOString(),
  // Start looking for signals from 1 hour ago (catches existing signals)
  last_updated: new Date(Date.now() - 60 * 60 * 1000).toISOString(),
}

async function loadState(): Promise<void> {
  try {
    const file = Bun.file(STATE_FILE)
    if (await file.exists()) {
      portfolio = await file.json()
      console.log(`[STATE] Loaded portfolio: $${portfolio.total_value.toFixed(2)} | ${portfolio.positions.length} positions | P&L: ${portfolio.total_pnl >= 0 ? '+' : ''}$${portfolio.total_pnl.toFixed(2)}`)
    } else {
      console.log(`[STATE] New portfolio initialized: $${INITIAL_CAPITAL.toFixed(2)}`)
    }
  } catch {
    console.log(`[STATE] Fresh portfolio: $${INITIAL_CAPITAL.toFixed(2)}`)
  }
}

async function saveState(): Promise<void> {
  portfolio.last_updated = new Date().toISOString()
  const dir = STATE_FILE.replace(/\/[^/]+$/, '')
  await Bun.write(STATE_FILE, JSON.stringify(portfolio, null, 2))
}

// ─── SurrealDB Queries ──────────────────────────────────────────────────────

async function querySignals(): Promise<SignalEvent[]> {
  // Get signals since last portfolio update
  const since = portfolio.last_updated
  const query = `SELECT event_type, timestamp, metadata FROM plexus_event WHERE source = 'polymarket.signals' AND timestamp > '${since}' AND event_type != 'signal.suggestion.hold' ORDER BY timestamp ASC`

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

  if (!response.ok) return []
  const results: any[] = await response.json()
  return results[0]?.result || []
}

async function getCurrentPrices(conditionIds: string[]): Promise<Map<string, Map<string, number>>> {
  if (conditionIds.length === 0) return new Map()

  // Get latest price for each condition_id:outcome from recent trades
  const ids = conditionIds.map(id => `'${id}'`).join(', ')
  const query = `SELECT metadata.condition_id, metadata.outcome, metadata.price, timestamp FROM plexus_event WHERE source = 'polymarket.live' AND metadata.condition_id IN [${ids}] ORDER BY timestamp DESC LIMIT 100`

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

  if (!response.ok) return new Map()
  const results: any[] = await response.json()
  const rows = results[0]?.result || []

  // Build condition_id → outcome → latest_price map
  const prices = new Map<string, Map<string, number>>()
  for (const row of rows) {
    const cid = row.metadata?.condition_id
    const outcome = row.metadata?.outcome
    const price = parseFloat(row.metadata?.price ?? '0')
    if (!cid || !outcome) continue

    if (!prices.has(cid)) prices.set(cid, new Map())
    const outcomeMap = prices.get(cid)!
    if (!outcomeMap.has(outcome)) {
      outcomeMap.set(outcome, price)  // First (most recent) wins
    }
  }

  return prices
}

// ─── Position Management ────────────────────────────────────────────────────

function openPosition(signal: SignalEvent): Position | null {
  const m = signal.metadata
  if (m.confidence < MIN_CONFIDENCE) return null
  if (portfolio.positions.length >= MAX_OPEN_POSITIONS) return null

  // Check for existing position in same market:outcome
  const existing = portfolio.positions.find(
    p => p.condition_id === m.condition_id && p.outcome === m.outcome
  )
  if (existing) return null  // Already have a position

  // Kelly-inspired position sizing
  const rawSize = portfolio.capital * m.confidence * POSITION_RISK
  const size = Math.min(rawSize, MAX_POSITION_SIZE, portfolio.capital * 0.10)
  if (size < 1) return null  // Too small

  const direction = m.signal.includes('buy') ? 'long' : 'short' as const
  const shares = direction === 'long'
    ? size / m.current_prob                    // Buy shares at current probability
    : size / (1 - m.current_prob)              // Short: profit if prob drops

  portfolio.capital -= size

  return {
    id: `pos-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    condition_id: m.condition_id,
    outcome: m.outcome,
    market_slug: m.market_slug,
    direction,
    entry_price: m.current_prob,
    current_price: m.current_prob,
    size_usd: size,
    shares,
    opened_at: new Date().toISOString(),
    signal_confidence: m.confidence,
    unrealized_pnl: 0,
    unrealized_pnl_pct: 0,
  }
}

function updatePositionPnL(pos: Position, currentPrice: number): void {
  pos.current_price = currentPrice

  if (pos.direction === 'long') {
    // Long: bought shares, profit if price goes up
    const currentValue = pos.shares * currentPrice
    pos.unrealized_pnl = currentValue - pos.size_usd
  } else {
    // Short: sold shares, profit if price goes down
    const currentValue = pos.shares * (1 - currentPrice)
    pos.unrealized_pnl = currentValue - pos.size_usd
  }

  pos.unrealized_pnl_pct = pos.size_usd > 0 ? pos.unrealized_pnl / pos.size_usd : 0
}

function shouldClosePosition(pos: Position): string | null {
  // 1. Profit target
  if (pos.unrealized_pnl_pct >= PROFIT_TARGET) {
    return `profit_target (+${(pos.unrealized_pnl_pct * 100).toFixed(1)}%)`
  }

  // 2. Stop loss
  if (pos.unrealized_pnl_pct <= STOP_LOSS) {
    return `stop_loss (${(pos.unrealized_pnl_pct * 100).toFixed(1)}%)`
  }

  // 3. Time decay
  const ageMs = Date.now() - new Date(pos.opened_at).getTime()
  const ageHours = ageMs / (1000 * 60 * 60)
  if (ageHours > TIME_DECAY_HOURS) {
    return `time_decay (${ageHours.toFixed(1)}h)`
  }

  return null
}

function closePosition(pos: Position, reason: string): void {
  portfolio.capital += pos.size_usd + pos.unrealized_pnl
  portfolio.realized_pnl += pos.unrealized_pnl
  portfolio.closed_count++

  if (pos.unrealized_pnl > 0) portfolio.win_count++
  else portfolio.loss_count++

  const idx = portfolio.positions.indexOf(pos)
  if (idx >= 0) portfolio.positions.splice(idx, 1)

  const pnlStr = pos.unrealized_pnl >= 0 ? `+$${pos.unrealized_pnl.toFixed(2)}` : `-$${Math.abs(pos.unrealized_pnl).toFixed(2)}`
  console.log(`  [CLOSE] ${pos.direction.toUpperCase()} ${pos.outcome}@${(pos.entry_price * 100).toFixed(1)}%→${(pos.current_price * 100).toFixed(1)}% | ${pnlStr} | ${reason}`)
}

// ─── Event Publisher ─────────────────────────────────────────────────────────

async function publishPortfolioEvent(eventType: string, payload: Record<string, any>): Promise<void> {
  if (DRY_RUN) return

  try {
    await fetch(PLEXUS_API, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        source: SOURCE,
        event_type: eventType,
        severity: 'info',
        message: payload.message || eventType,
        payload,
      }),
    })
  } catch { /* best effort */ }
}

// ─── Main Loop ──────────────────────────────────────────────────────────────

async function runCycle(): Promise<void> {
  // 1. Get new signals
  const signals = await querySignals()

  // 2. Process new signals → open positions
  let opened = 0
  for (const signal of signals) {
    const isBuy = signal.event_type.includes('buy') || signal.event_type.includes('strongbuy')
    const isSell = signal.event_type.includes('sell') || signal.event_type.includes('strongsell')

    if (!isBuy && !isSell) continue

    // Check for signal reversal (close opposing position)
    const opposing = portfolio.positions.find(
      p => p.condition_id === signal.metadata.condition_id &&
           p.outcome === signal.metadata.outcome &&
           ((isBuy && p.direction === 'short') || (isSell && p.direction === 'long'))
    )
    if (opposing) {
      updatePositionPnL(opposing, signal.metadata.current_prob)
      closePosition(opposing, 'signal_reversal')
      await publishPortfolioEvent('portfolio.position.closed', {
        position_id: opposing.id,
        reason: 'signal_reversal',
        pnl: opposing.unrealized_pnl,
        message: `Closed ${opposing.direction} on signal reversal`,
      })
    }

    const pos = openPosition(signal)
    if (pos) {
      portfolio.positions.push(pos)
      opened++
      console.log(`  [OPEN]  ${pos.direction.toUpperCase()} ${pos.outcome}@${(pos.entry_price * 100).toFixed(1)}% | $${pos.size_usd.toFixed(2)} (conf: ${(pos.signal_confidence * 100).toFixed(0)}%)`)
      await publishPortfolioEvent('portfolio.position.opened', {
        position_id: pos.id,
        direction: pos.direction,
        outcome: pos.outcome,
        entry_price: pos.entry_price,
        size_usd: pos.size_usd,
        message: `Opened ${pos.direction} ${pos.outcome}@${(pos.entry_price * 100).toFixed(1)}%`,
      })
    }
  }

  // 3. Update existing positions with current prices
  if (portfolio.positions.length > 0) {
    const conditionIds = [...new Set(portfolio.positions.map(p => p.condition_id))]
    const prices = await getCurrentPrices(conditionIds)

    for (const pos of [...portfolio.positions]) {
      const marketPrices = prices.get(pos.condition_id)
      const currentPrice = marketPrices?.get(pos.outcome)

      if (currentPrice !== undefined) {
        updatePositionPnL(pos, currentPrice)

        const closeReason = shouldClosePosition(pos)
        if (closeReason) {
          closePosition(pos, closeReason)
          await publishPortfolioEvent('portfolio.position.closed', {
            position_id: pos.id,
            reason: closeReason,
            pnl: pos.unrealized_pnl,
            message: `Closed: ${closeReason}`,
          })
        }
      }
    }
  }

  // 4. Calculate total value
  const positionsValue = portfolio.positions.reduce((sum, p) => sum + p.size_usd + p.unrealized_pnl, 0)
  portfolio.total_value = portfolio.capital + positionsValue
  portfolio.total_pnl = portfolio.total_value - portfolio.initial_capital

  // 5. Save state
  await saveState()

  // 6. Print status
  printStatus(signals.length, opened)
}

function printStatus(newSignals: number, opened: number): void {
  const pnlStr = portfolio.total_pnl >= 0
    ? `+$${portfolio.total_pnl.toFixed(2)}`
    : `-$${Math.abs(portfolio.total_pnl).toFixed(2)}`
  const pnlPct = ((portfolio.total_pnl / portfolio.initial_capital) * 100).toFixed(2)
  const winRate = portfolio.closed_count > 0
    ? ((portfolio.win_count / portfolio.closed_count) * 100).toFixed(0)
    : 'N/A'

  console.log(`\n[PORTFOLIO] ${new Date().toISOString()}`)
  console.log(`  Value:    $${portfolio.total_value.toFixed(2)} (${pnlStr}, ${pnlPct}%)`)
  console.log(`  Cash:     $${portfolio.capital.toFixed(2)}`)
  console.log(`  Positions: ${portfolio.positions.length}/${MAX_OPEN_POSITIONS}`)
  console.log(`  Trades:   ${portfolio.closed_count} closed (${winRate}% win rate)`)
  console.log(`  Signals:  ${newSignals} new, ${opened} acted on`)

  if (portfolio.positions.length > 0) {
    console.log(`  ── Open Positions ──`)
    for (const p of portfolio.positions) {
      const dir = p.direction === 'long' ? '▲' : '▼'
      const pnl = p.unrealized_pnl >= 0 ? `+$${p.unrealized_pnl.toFixed(2)}` : `-$${Math.abs(p.unrealized_pnl).toFixed(2)}`
      const pnlPct = (p.unrealized_pnl_pct * 100).toFixed(1)
      console.log(`  ${dir} ${p.outcome.padEnd(6)}@${(p.entry_price * 100).toFixed(1)}%→${(p.current_price * 100).toFixed(1)}% | $${p.size_usd.toFixed(2)} | ${pnl} (${pnlPct}%)`)
    }
  }
}

// ─── Entry ──────────────────────────────────────────────────────────────────

// Ensure state directory exists
const stateDir = STATE_FILE.replace(/\/[^/]+$/, '')
await Bun.write(`${stateDir}/.keep`, '')

console.log(`[PORTFOLIO TRACKER] Polymarket Virtual Portfolio`)
console.log(`[CONFIG] Capital: $${INITIAL_CAPITAL} | Cycle: ${CYCLE_SECONDS}s | Max position: $${MAX_POSITION_SIZE}`)
console.log(`[CONFIG] Min confidence: ${(MIN_CONFIDENCE * 100).toFixed(0)}% | Max positions: ${MAX_OPEN_POSITIONS}`)
console.log(`[CONFIG] Profit target: +${(PROFIT_TARGET * 100).toFixed(0)}% | Stop loss: ${(STOP_LOSS * 100).toFixed(0)}%`)
if (DRY_RUN) console.log('[CONFIG] DRY RUN — no events published')

await loadState()

if (STATUS_ONLY) {
  printStatus(0, 0)
  process.exit(0)
}

// Initial cycle
await runCycle()

if (!STATUS_ONLY) {
  setInterval(async () => {
    try {
      await runCycle()
    } catch (err: any) {
      console.error(`[ERROR] Cycle failed: ${err.message}`)
    }
  }, CYCLE_SECONDS * 1000)

  process.on('SIGINT', async () => {
    await saveState()
    console.log(`\n[SHUTDOWN] Final value: $${portfolio.total_value.toFixed(2)} | P&L: ${portfolio.total_pnl >= 0 ? '+' : ''}$${portfolio.total_pnl.toFixed(2)}`)
    process.exit(0)
  })
}
