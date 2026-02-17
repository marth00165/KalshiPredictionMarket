#!/usr/bin/env bun
/**
 * Market Feed → Substrate Adapter
 *
 * Connects to Coinbase Advanced Trade WebSocket, receives BTC-USD trades,
 * and publishes each as a substrate Synapse event via `nabi events publish`.
 *
 * This is the proof: the exact same pipeline that replayed 113 football plays
 * now ingests real-time BTC trades. NabiOS is domain-agnostic.
 *
 * Usage:
 *   bun run adapter.ts                    # Default: market_trades, 1 tick/sec
 *   bun run adapter.ts --channel ticker   # 24h ticker updates (~4/sec)
 *   bun run adapter.ts --throttle 0       # No throttle (full firehose)
 *   bun run adapter.ts --dry-run          # Print events without publishing
 *   bun run adapter.ts --symbol ETH-USD   # Different pair
 *   bun run adapter.ts --max 50           # Stop after 50 events
 *
 * Exchanges:
 *   --exchange coinbase     (default, US-native, BTC-USD)
 *   --exchange binance      (non-US, BTCUSDT — blocked in US)
 *   --exchange polymarket   (prediction markets, all trades, no auth)
 *   --exchange kalshi       (prediction markets, REST polling, no auth for read)
 *
 * Polymarket-specific flags:
 *   --market-filter <slug>  # Filter to specific event (e.g. "us-presidential-election")
 *
 * Kalshi-specific flags:
 *   --kalshi-poll <ms>      # Poll interval (default: 60000 = 1 minute)
 *   --kalshi-limit <n>      # Markets per poll (default: 50)
 *   --kalshi-category <cat> # Filter by category (e.g. "politics", "economics")
 *
 * Polymarket examples:
 *   bun run adapter.ts --exchange polymarket                    # All prediction market trades
 *   bun run adapter.ts --exchange polymarket --throttle 500     # 2 events/sec
 *   bun run adapter.ts --exchange polymarket --market-filter us-presidential-election
 *   bun run adapter.ts --exchange polymarket --dry-run --max 20 # Preview 20 trades
 *
 * Kalshi examples:
 *   bun run adapter.ts --exchange kalshi                        # 50 markets, 1min polls
 *   bun run adapter.ts --exchange kalshi --kalshi-limit 20      # Fewer markets (faster polls)
 *   bun run adapter.ts --exchange kalshi --kalshi-poll 30000    # 30s polls (more aggressive)
 *   bun run adapter.ts --exchange kalshi --kalshi-category politics --dry-run
 *   bun run adapter.ts --exchange kalshi --dry-run --max 100    # Preview first 100 events
 *
 * Substrate Mapping:
 *   trade tick    → Synapse (atomic event)
 *   price spike   → severity: warning/critical
 *   volume burst  → severity: critical
 *   connection    → source: market-feed:META
 */

import { $ } from 'bun'

// ─── Config ─────────────────────────────────────────────────────────────────

const args = process.argv.slice(2)
function getArg(name: string, fallback: string): string {
  const idx = args.indexOf(`--${name}`)
  return idx >= 0 && args[idx + 1] ? args[idx + 1] : fallback
}
const hasFlag = (name: string) => args.includes(`--${name}`)

const EXCHANGE = getArg('exchange', 'coinbase')
const SYMBOL = getArg('symbol', 'BTC-USD')
const CHANNEL = getArg('channel', 'market_trades')  // market_trades | ticker
const THROTTLE_MS = parseInt(getArg('throttle', '1000'))
const DRY_RUN = hasFlag('dry-run')
const MAX_EVENTS = parseInt(getArg('max', '0'))
const MARKET_FILTER = getArg('market-filter', '')  // Polymarket: event_slug filter

// Kalshi-specific flags
const KALSHI_POLL_MS = parseInt(getArg('kalshi-poll', '60000'))  // 1min default
const KALSHI_LIMIT = parseInt(getArg('kalshi-limit', '20'))      // Markets per poll (keep low to respect rate limits)
const KALSHI_CATEGORY = getArg('kalshi-category', '')            // Optional category filter
const KALSHI_API_URL = 'https://api.elections.kalshi.com/trade-api/v2'

// Source name for the substrate (normalized)
const SOURCE_TRADE = EXCHANGE === 'polymarket'
  ? 'polymarket.live'
  : EXCHANGE === 'kalshi'
  ? 'kalshi.live'
  : `market-feed:${SYMBOL.replace('-', '')}`
const SOURCE_META = EXCHANGE === 'polymarket'
  ? 'polymarket.meta'
  : EXCHANGE === 'kalshi'
  ? 'kalshi.meta'
  : 'market-feed:META'

const WS_ENDPOINTS: Record<string, string> = {
  coinbase: 'wss://advanced-trade-ws.coinbase.com',
  binance: 'wss://stream.binance.com:9443/ws',
  polymarket: 'wss://ws-live-data.polymarket.com',
  // Kalshi: REST polling, no WebSocket — handled separately
}

// ─── State ──────────────────────────────────────────────────────────────────

let seq = 0
let lastPublishTime = 0
let eventCount = 0
let errorCount = 0
let lastPrice = 0
let sessionId = crypto.randomUUID()
let priceWindow: number[] = []
let volumeWindow: number[] = []
const WINDOW_SIZE = 60
let reconnectAttempts = 0

// ─── Severity Engine ────────────────────────────────────────────────────────

interface SeverityResult {
  level: 'info' | 'warning' | 'critical'
  reason?: string
}

function computeSeverity(price: number, quantity: number): SeverityResult {
  const priceDelta = lastPrice > 0
    ? Math.abs((price - lastPrice) / lastPrice) * 100
    : 0

  const avgVol = volumeWindow.length > 0
    ? volumeWindow.reduce((a, b) => a + b, 0) / volumeWindow.length
    : quantity
  const stdVol = volumeWindow.length > 2
    ? Math.sqrt(volumeWindow.reduce((sum, v) => sum + (v - avgVol) ** 2, 0) / volumeWindow.length)
    : avgVol * 0.5
  const volZScore = stdVol > 0 ? (quantity - avgVol) / stdVol : 0

  if (priceDelta > 0.5 || volZScore > 3) {
    return {
      level: 'critical',
      reason: priceDelta > 0.5
        ? `price_move_${priceDelta.toFixed(2)}pct`
        : `volume_spike_${volZScore.toFixed(1)}sigma`,
    }
  }

  if (priceDelta > 0.1 || volZScore > 2) {
    return {
      level: 'warning',
      reason: priceDelta > 0.1
        ? `price_shift_${priceDelta.toFixed(3)}pct`
        : `volume_elevated_${volZScore.toFixed(1)}sigma`,
    }
  }

  return { level: 'info' }
}

// ─── Event Publisher ────────────────────────────────────────────────────────

/**
 * ADR-0067: Envelope/Payload Separation Pattern
 *
 * Posts full payload to plexus-bridge HTTP API, which:
 * 1. Writes full payload to SurrealDB (durable storage)
 * 2. Publishes lightweight envelope to NATS (~400 bytes vs 5KB+)
 * 3. Returns record_id for reference
 *
 * Before: 5KB+ events → NATS → congestion
 * After: 5KB → SurrealDB, 400 bytes → NATS (90%+ reduction)
 */
async function publishEvent(
  source: string,
  eventType: string,
  severity: string,
  metadata: Record<string, any>,
  description: string,
): Promise<boolean> {
  if (DRY_RUN) {
    const sev = severity.toUpperCase().padEnd(8)
    console.log(`[DRY] ${sev} ${source} | ${eventType} | ${description}`)
    return true
  }

  try {
    // ADR-0067: Full payload goes to HTTP API
    const payload = {
      source,
      event_type: eventType,
      severity,
      message: description,
      payload: metadata,  // Full metadata stored in SurrealDB
    }

    const response = await fetch('http://localhost:5381/events/full', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    if (!response.ok) {
      errorCount++
      const errorText = await response.text().catch(() => 'Unknown error')
      console.error(`[HTTP ERROR] ${response.status}: ${errorText.slice(0, 200)}`)
      return false
    }

    const result = await response.json()
    // result contains: { record_id, envelope }
    // envelope is what was published to NATS (~400 bytes)

    return true
  } catch (err: any) {
    errorCount++
    console.error(`[PUBLISH EXCEPTION] ${err.message?.slice(0, 200)}`)
    return false
  }
}

// ─── Coinbase Handlers ──────────────────────────────────────────────────────

function handleCoinbaseTrade(trade: any) {
  const now = Date.now()
  if (THROTTLE_MS > 0 && (now - lastPublishTime) < THROTTLE_MS) return

  seq++
  const price = parseFloat(trade.price)
  const size = parseFloat(trade.size)
  const side = trade.side?.toLowerCase() || 'unknown'
  const tradeTime = trade.time || new Date().toISOString()
  const receiveTime = new Date(now).toISOString()

  const { level, reason } = computeSeverity(price, size)

  // Update rolling windows
  priceWindow.push(price)
  volumeWindow.push(size)
  if (priceWindow.length > WINDOW_SIZE) priceWindow.shift()
  if (volumeWindow.length > WINDOW_SIZE) volumeWindow.shift()

  const priceDelta = lastPrice > 0 ? ((price - lastPrice) / lastPrice * 100) : 0
  lastPrice = price
  lastPublishTime = now
  eventCount++

  const metadata: Record<string, any> = {
    seq,
    symbol: SYMBOL,
    exchange: EXCHANGE,
    event_type: 'trade',
    price: trade.price,
    quantity: trade.size,
    side,
    trade_id: parseInt(trade.trade_id) || 0,
    timestamp: tradeTime,
    receive_time: receiveTime,
    session_id: sessionId,
  }

  if (reason) {
    metadata.derived = {
      price_delta_pct: parseFloat(priceDelta.toFixed(4)),
      severity_reason: reason,
    }
  }

  const priceStr = parseFloat(trade.price).toFixed(2)
  const desc = `${SYMBOL} $${priceStr} x${trade.size} (${side.toUpperCase()})${reason ? ` [${reason}]` : ''}`

  publishEvent(SOURCE_TRADE, `market.trade.${side}`, level, metadata, desc)

  if (MAX_EVENTS > 0 && eventCount >= MAX_EVENTS) {
    console.log(`\n[DONE] Reached max events (${MAX_EVENTS}). Closing.`)
    process.exit(0)
  }
}

function handleCoinbaseTicker(ticker: any) {
  const now = Date.now()
  if (THROTTLE_MS > 0 && (now - lastPublishTime) < THROTTLE_MS) return

  seq++
  const price = parseFloat(ticker.price)
  const volume = parseFloat(ticker.volume_24_h || '0')
  const { level, reason } = computeSeverity(price, volume)

  lastPrice = price
  lastPublishTime = now
  eventCount++

  const metadata: Record<string, any> = {
    seq,
    symbol: SYMBOL,
    exchange: EXCHANGE,
    event_type: 'ticker',
    price: ticker.price,
    quantity: ticker.volume_24_h,
    side: 'unknown',
    timestamp: new Date().toISOString(),
    receive_time: new Date(now).toISOString(),
    session_id: sessionId,
    ticker: {
      high: ticker.high_24_h,
      low: ticker.low_24_h,
      volume: ticker.volume_24_h,
      best_bid: ticker.best_bid,
      best_ask: ticker.best_ask,
    },
  }

  if (reason) {
    metadata.derived = { severity_reason: reason }
  }

  const priceStr = parseFloat(ticker.price).toFixed(2)
  const desc = `${SYMBOL} $${priceStr} bid:${ticker.best_bid} ask:${ticker.best_ask}`

  publishEvent(SOURCE_TRADE, 'market.ticker.update', level, metadata, desc)

  if (MAX_EVENTS > 0 && eventCount >= MAX_EVENTS) {
    console.log(`\n[DONE] Reached max events (${MAX_EVENTS}). Closing.`)
    process.exit(0)
  }
}

// ─── Polymarket State ────────────────────────────────────────────────────────

// Per-market price tracking for Polymarket (many markets, each 0-1)
const marketPrices: Map<string, number> = new Map()
const marketVolumes: Map<string, number[]> = new Map()
const marketNames: Map<string, string> = new Map()  // conditionId → slug or label

// ─── Polymarket Handlers ─────────────────────────────────────────────────────

function handlePolymarketTrade(payload: any) {
  const now = Date.now()
  if (THROTTLE_MS > 0 && (now - lastPublishTime) < THROTTLE_MS) return

  seq++

  // Extract trade fields from Polymarket activity/trades payload
  const price = parseFloat(payload.price ?? payload.outcome_price ?? '0')
  const size = parseFloat(payload.size ?? payload.amount ?? '0')
  const side = (payload.side ?? 'unknown').toUpperCase()
  const conditionId = payload.condition_id ?? payload.conditionId ?? 'unknown'
  const outcome = payload.outcome ?? payload.asset_id ?? 'unknown'
  const marketSlug = payload.market_slug ?? payload.event_slug ?? conditionId
  const tradeId = payload.id ?? payload.trade_id ?? `${conditionId}-${seq}`

  // Track market name for enrichment
  if (marketSlug !== conditionId) {
    marketNames.set(conditionId, marketSlug)
  }

  // Track per-asset (conditionId:outcome), NOT per-conditionId alone.
  // Prediction markets have complementary outcomes (YES+NO≈100%),
  // so tracking by conditionId would create false severity spikes.
  const assetKey = `${conditionId}:${outcome}`

  // Per-asset price delta (probability 0-1)
  const lastMarketPrice = marketPrices.get(assetKey) ?? price
  const priceDelta = Math.abs(price - lastMarketPrice)
  marketPrices.set(assetKey, price)

  // Per-asset volume window (for z-score severity)
  let volumes = marketVolumes.get(assetKey)
  if (!volumes) {
    volumes = []
    marketVolumes.set(assetKey, volumes)
  }
  volumes.push(size)
  if (volumes.length > WINDOW_SIZE) volumes.shift()

  // Compute severity using probability-aware thresholds
  const { level, reason } = computePolymarketSeverity(price, size, priceDelta, volumes)

  // Also update global windows for cross-market analysis
  priceWindow.push(price)
  volumeWindow.push(size)
  if (priceWindow.length > WINDOW_SIZE) priceWindow.shift()
  if (volumeWindow.length > WINDOW_SIZE) volumeWindow.shift()

  lastPrice = price
  lastPublishTime = now
  eventCount++

  const metadata: Record<string, any> = {
    seq,
    exchange: 'polymarket',
    event_type: 'trade',
    market_type: 'prediction',
    condition_id: conditionId,
    market_slug: marketSlug,
    outcome,
    price: price.toFixed(4),
    probability_pct: (price * 100).toFixed(1),
    size: size.toFixed(2),
    side,
    trade_id: tradeId,
    timestamp: payload.timestamp ? new Date(payload.timestamp * 1000).toISOString() : new Date().toISOString(),
    receive_time: new Date(now).toISOString(),
    session_id: sessionId,
  }

  if (reason) {
    metadata.derived = {
      price_delta: priceDelta.toFixed(4),
      severity_reason: reason,
    }
  }

  const probStr = (price * 100).toFixed(1)
  const slugShort = marketSlug.length > 40 ? marketSlug.slice(0, 40) + '...' : marketSlug
  const desc = `${slugShort} ${outcome}:${probStr}% x${size.toFixed(1)} (${side})${reason ? ` [${reason}]` : ''}`

  publishEvent(SOURCE_TRADE, `polymarket.trade.${side.toLowerCase()}`, level, metadata, desc)

  if (MAX_EVENTS > 0 && eventCount >= MAX_EVENTS) {
    console.log(`\n[DONE] Reached max events (${MAX_EVENTS}). Closing.`)
    process.exit(0)
  }
}

/**
 * Severity engine for prediction markets (probability-priced 0-1)
 * Different from financial markets:
 * - "Price" is probability (0.00-1.00), not USD
 * - A 5-cent move (0.50→0.55) is significant in prediction markets
 * - Volume spikes still use z-score (universal)
 */
function computePolymarketSeverity(
  price: number,
  size: number,
  priceDelta: number,
  volumes: number[],
): SeverityResult {
  // Volume z-score (same as financial)
  const avgVol = volumes.length > 0
    ? volumes.reduce((a, b) => a + b, 0) / volumes.length
    : size
  const stdVol = volumes.length > 2
    ? Math.sqrt(volumes.reduce((sum, v) => sum + (v - avgVol) ** 2, 0) / volumes.length)
    : avgVol * 0.5
  const volZScore = stdVol > 0 ? (size - avgVol) / stdVol : 0

  // Probability-aware thresholds:
  // In prediction markets, 5 cents = meaningful, 10 cents = huge
  if (priceDelta > 0.10 || volZScore > 3) {
    return {
      level: 'critical',
      reason: priceDelta > 0.10
        ? `prob_shift_${(priceDelta * 100).toFixed(1)}pct`
        : `volume_spike_${volZScore.toFixed(1)}sigma`,
    }
  }

  if (priceDelta > 0.03 || volZScore > 2) {
    return {
      level: 'warning',
      reason: priceDelta > 0.03
        ? `prob_move_${(priceDelta * 100).toFixed(1)}pct`
        : `volume_elevated_${volZScore.toFixed(1)}sigma`,
    }
  }

  return { level: 'info' }
}

// ─── Kalshi REST Polling ─────────────────────────────────────────────────────

// Per-market state for Kalshi (tracks price changes between polls)
const kalshiPrices: Map<string, number> = new Map()
const kalshiVolumes: Map<string, number[]> = new Map()

// Excluded ticker prefixes (auto-generated bundle markets, not useful)
const KALSHI_EXCLUDED_PREFIXES = ['KXMVESPORTSMULTIGAMEEXTENDED']

interface KalshiMarket {
  ticker: string
  title: string
  subtitle: string
  status: string
  category: string
  volume: number
  open_interest: number
  expiration_time: string
  yes_sub_title?: string
  no_sub_title?: string
}

async function kalshiFetch(url: string, retries = 3): Promise<Response | null> {
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const resp = await fetch(url, {
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'NabiOS-MarketAdapter/1.0',
        },
      })
      if (resp.status === 429) {
        const retryAfter = parseInt(resp.headers.get('Retry-After') ?? '') || (2 ** attempt * 2)
        console.warn(`[KALSHI] 429 rate-limited, retry in ${retryAfter}s (attempt ${attempt + 1}/${retries})`)
        await Bun.sleep(retryAfter * 1000)
        continue
      }
      return resp
    } catch (err: any) {
      if (attempt < retries - 1) {
        const delay = 2 ** attempt * 1000
        console.warn(`[KALSHI] Fetch error, retry in ${delay}ms: ${err.message?.slice(0, 100)}`)
        await Bun.sleep(delay)
      } else {
        console.error(`[KALSHI] Fetch failed after ${retries} attempts: ${err.message?.slice(0, 200)}`)
      }
    }
  }
  return null
}

async function fetchKalshiMarkets(): Promise<KalshiMarket[]> {
  const markets: KalshiMarket[] = []
  let cursor: string | undefined

  while (markets.length < KALSHI_LIMIT) {
    const params = new URLSearchParams({
      limit: String(Math.min(100, KALSHI_LIMIT - markets.length)),
      status: 'open',
    })
    if (cursor) params.set('cursor', cursor)
    if (KALSHI_CATEGORY) params.set('category', KALSHI_CATEGORY)

    const url = `${KALSHI_API_URL}/markets?${params}`
    const resp = await kalshiFetch(url)
    if (!resp || !resp.ok) {
      if (resp) console.error(`[KALSHI] API error: ${resp.status} ${resp.statusText}`)
      break
    }
    const data = await resp.json() as { markets?: KalshiMarket[], cursor?: string }
    const batch = data.markets ?? []
    if (batch.length === 0) break

    for (const m of batch) {
      if (KALSHI_EXCLUDED_PREFIXES.some(pfx => m.ticker.startsWith(pfx))) continue
      markets.push(m)
    }

    cursor = data.cursor
    if (!cursor) break
  }
  return markets
}

async function fetchKalshiPrice(ticker: string): Promise<number> {
  const resp = await kalshiFetch(`${KALSHI_API_URL}/markets/${ticker}/orderbook`, 2)
  if (!resp || !resp.ok) return 0.5
  try {
    const data = await resp.json() as { orderbook?: { yes?: number[][], no?: number[][] } }
    const yesBids = data.orderbook?.yes ?? []
    if (yesBids.length > 0 && yesBids[0].length > 0) {
      return yesBids[0][0] / 100  // Cents → probability (0-1)
    }
  } catch { /* parse error → default */ }
  return 0.5
}

async function pollKalshiOnce() {
  const start = Date.now()
  const markets = await fetchKalshiMarkets()
  console.log(`[KALSHI] Fetched ${markets.length} markets in ${Date.now() - start}ms`)

  let priced = 0
  for (const market of markets) {
    // Rate-limit orderbook fetches to avoid 429s
    const yesPrice = await fetchKalshiPrice(market.ticker)
    priced++

    // Throttle: ~150ms between orderbook calls
    if (priced < markets.length) await Bun.sleep(150)

    seq++
    eventCount++

    const noPrice = 1 - yesPrice
    const volume = market.volume ?? 0
    const openInterest = market.open_interest ?? 0

    // Per-market price tracking
    const lastPrice = kalshiPrices.get(market.ticker) ?? yesPrice
    const priceDelta = Math.abs(yesPrice - lastPrice)
    kalshiPrices.set(market.ticker, yesPrice)

    // Per-market volume window for severity
    let vols = kalshiVolumes.get(market.ticker)
    if (!vols) { vols = []; kalshiVolumes.set(market.ticker, vols) }
    vols.push(volume)
    if (vols.length > WINDOW_SIZE) vols.shift()

    // Reuse prediction market severity (same 0-1 probability scale)
    const { level, reason } = computePolymarketSeverity(yesPrice, volume, priceDelta, vols)

    const metadata: Record<string, any> = {
      seq,
      exchange: 'kalshi',
      event_type: 'market.snapshot',
      market_type: 'prediction',
      ticker: market.ticker,
      title: market.title,
      category: market.category || 'other',
      yes_price: yesPrice.toFixed(4),
      no_price: noPrice.toFixed(4),
      probability_pct: (yesPrice * 100).toFixed(1),
      volume,
      open_interest: openInterest,
      expiration_time: market.expiration_time,
      price_delta: priceDelta.toFixed(4),
      timestamp: new Date().toISOString(),
      receive_time: new Date().toISOString(),
      session_id: sessionId,
    }

    if (reason) metadata.derived = { severity_reason: reason }

    const probStr = (yesPrice * 100).toFixed(1)
    const desc = `${market.ticker} YES:${probStr}% vol:${volume} OI:${openInterest}`

    publishEvent('kalshi.live', 'market.snapshot', level, metadata, desc)
  }

  console.log(`[KALSHI] Published ${markets.length} snapshots (${priced} priced)`)
}

function startKalshiPolling() {
  console.log(`[ADAPTER] Kalshi REST polling mode`)
  console.log(`[ADAPTER] Source: ${SOURCE_TRADE} | Poll: ${KALSHI_POLL_MS}ms | Limit: ${KALSHI_LIMIT}`)
  console.log(`[ADAPTER] Session: ${sessionId}`)
  if (KALSHI_CATEGORY) console.log(`[ADAPTER] Category filter: ${KALSHI_CATEGORY}`)
  if (DRY_RUN) console.log(`[ADAPTER] DRY RUN — events printed, not published`)
  console.log('')

  // Initial connection event
  publishEvent('kalshi.meta', 'market.connection.open', 'info', {
    exchange: 'kalshi',
    mode: 'rest-polling',
    poll_interval_ms: KALSHI_POLL_MS,
    markets_limit: KALSHI_LIMIT,
    category_filter: KALSHI_CATEGORY || undefined,
    session_id: sessionId,
  }, `Kalshi REST adapter started: ${KALSHI_LIMIT} markets, ${KALSHI_POLL_MS}ms interval`)

  // First poll immediately
  pollKalshiOnce()

  // Then poll on interval
  const pollInterval = setInterval(pollKalshiOnce, KALSHI_POLL_MS)

  // Status reporter every 30s
  const statusInterval = setInterval(() => {
    if (eventCount === 0) return
    console.log(
      `[STATUS] ${new Date().toISOString()} | events: ${eventCount} | errors: ${errorCount} | ` +
      `markets: ${kalshiPrices.size} | seq: ${seq}`,
    )
  }, 30_000)

  // Graceful shutdown
  const shutdown = () => {
    clearInterval(pollInterval)
    clearInterval(statusInterval)
    console.log(`\n[SHUTDOWN] ${eventCount} events, ${errorCount} errors`)
    publishEvent('kalshi.meta', 'market.connection.shutdown', 'info', {
      exchange: 'kalshi',
      session_id: sessionId,
      events_published: eventCount,
      errors: errorCount,
      markets_tracked: kalshiPrices.size,
    }, `Kalshi adapter shutdown: ${eventCount} events, ${kalshiPrices.size} markets`)
    setTimeout(() => process.exit(0), 500)
  }

  process.on('SIGINT', shutdown)
  process.on('SIGTERM', shutdown)
}

// ─── WebSocket Connection ───────────────────────────────────────────────────

function connect() {
  // Kalshi uses REST polling, not WebSocket
  if (EXCHANGE === 'kalshi') {
    startKalshiPolling()
    return
  }

  const wsUrl = WS_ENDPOINTS[EXCHANGE]
  if (!wsUrl) {
    console.error(`[FATAL] Unknown exchange: ${EXCHANGE}`)
    process.exit(1)
  }

  console.log(`[ADAPTER] Connecting to ${EXCHANGE} WebSocket`)
  console.log(`[ADAPTER] Source: ${SOURCE_TRADE} | Channel: ${CHANNEL} | Throttle: ${THROTTLE_MS}ms`)
  console.log(`[ADAPTER] Session: ${sessionId}`)
  if (DRY_RUN) console.log(`[ADAPTER] DRY RUN — events printed, not published`)
  if (MAX_EVENTS > 0) console.log(`[ADAPTER] Max events: ${MAX_EVENTS}`)
  console.log('')

  const connectionDesc = EXCHANGE === 'polymarket'
    ? `Polymarket live trades${MARKET_FILTER ? ` (filter: ${MARKET_FILTER})` : ' (all markets)'}`
    : `Market feed adapter connected: ${SYMBOL}@${CHANNEL}`

  publishEvent(SOURCE_META, 'market.connection.open', 'info', {
    symbol: EXCHANGE === 'polymarket' ? 'PREDICTION_MARKETS' : SYMBOL,
    exchange: EXCHANGE,
    channel: EXCHANGE === 'polymarket' ? 'activity/trades' : CHANNEL,
    session_id: sessionId,
    throttle_ms: THROTTLE_MS,
    market_filter: MARKET_FILTER || undefined,
  }, connectionDesc)

  const ws = new WebSocket(wsUrl)

  ws.onopen = () => {
    console.log(`[CONNECTED] ${new Date().toISOString()} — streaming ${EXCHANGE === 'polymarket' ? 'prediction markets' : SYMBOL}`)
    reconnectAttempts = 0

    if (EXCHANGE === 'coinbase') {
      ws.send(JSON.stringify({
        type: 'subscribe',
        product_ids: [SYMBOL],
        channel: CHANNEL,
      }))
    } else if (EXCHANGE === 'polymarket') {
      // Polymarket real-time data client subscription format
      const subscription: any = {
        topic: 'activity',
        type: 'trades',
      }
      if (MARKET_FILTER) {
        subscription.filters = JSON.stringify({ event_slug: MARKET_FILTER })
      }
      ws.send(JSON.stringify({
        action: 'subscribe',
        subscriptions: [subscription],
      }))
      console.log(`[SUBSCRIBE] topic=activity type=trades${MARKET_FILTER ? ` filter=${MARKET_FILTER}` : ' (all markets)'}`)
    }
  }

  ws.onmessage = (event) => {
    try {
      const raw = event.data as string
      // Skip non-JSON messages (pings, connection acks)
      if (!raw.startsWith('{') && !raw.startsWith('[')) return
      const msg = JSON.parse(raw)

      if (EXCHANGE === 'coinbase') {
        if (msg.channel === 'market_trades' && msg.events) {
          for (const evt of msg.events) {
            if (evt.trades) {
              for (const trade of evt.trades) {
                handleCoinbaseTrade(trade)
              }
            }
          }
        } else if (msg.channel === 'ticker' && msg.events) {
          for (const evt of msg.events) {
            if (evt.tickers) {
              for (const ticker of evt.tickers) {
                handleCoinbaseTicker(ticker)
              }
            }
          }
        }
      } else if (EXCHANGE === 'polymarket') {
        // Polymarket messages have { topic, type, timestamp, payload }
        if (msg.payload) {
          if (msg.topic === 'activity' && msg.type === 'trades') {
            // payload can be a single trade or array
            const trades = Array.isArray(msg.payload) ? msg.payload : [msg.payload]
            for (const trade of trades) {
              handlePolymarketTrade(trade)
            }
          }
        }
        // Ignore connection acks, pings, and non-payload messages
      }
    } catch (err: any) {
      console.error(`[PARSE ERROR] ${err.message}`)
    }
  }

  ws.onerror = () => {
    console.error(`[WS ERROR] ${new Date().toISOString()}`)
    publishEvent(SOURCE_META, 'market.connection.error', 'error', {
      symbol: SYMBOL,
      exchange: EXCHANGE,
      session_id: sessionId,
    }, `Market feed WebSocket error: ${SYMBOL}`)
  }

  ws.onclose = (event) => {
    reconnectAttempts++
    console.log(`[DISCONNECTED] code: ${event.code}, reason: ${event.reason}`)

    publishEvent(SOURCE_META, 'market.connection.close', 'warning', {
      symbol: SYMBOL,
      exchange: EXCHANGE,
      session_id: sessionId,
      close_code: event.code,
      events_published: eventCount,
      errors: errorCount,
    }, `Market feed disconnected after ${eventCount} events`)

    // Exponential backoff with max 30s
    const delay = Math.min(30000, 1000 * (2 ** Math.min(reconnectAttempts, 5)))
    console.log(`[RECONNECT] Attempt ${reconnectAttempts} in ${delay}ms...`)
    setTimeout(() => {
      sessionId = crypto.randomUUID()
      connect()
    }, delay)
  }

  // Status reporter every 30s
  const statusInterval = setInterval(() => {
    if (eventCount === 0) return
    if (EXCHANGE === 'polymarket') {
      console.log(
        `[STATUS] ${new Date().toISOString()} | events: ${eventCount} | errors: ${errorCount} | ` +
        `markets: ${marketPrices.size} | seq: ${seq}`,
      )
    } else {
      const avgPrice = priceWindow.length > 0
        ? (priceWindow.reduce((a, b) => a + b, 0) / priceWindow.length).toFixed(2)
        : 'N/A'
      console.log(
        `[STATUS] ${new Date().toISOString()} | events: ${eventCount} | errors: ${errorCount} | ` +
        `last: $${lastPrice.toFixed(2)} | avg(${WINDOW_SIZE}): $${avgPrice} | seq: ${seq}`,
      )
    }
  }, 30_000)

  // Graceful shutdown
  const shutdown = () => {
    clearInterval(statusInterval)
    console.log(`\n[SHUTDOWN] ${eventCount} events, ${errorCount} errors, final: $${lastPrice.toFixed(2)}`)
    publishEvent(SOURCE_META, 'market.connection.shutdown', 'info', {
      symbol: SYMBOL,
      exchange: EXCHANGE,
      session_id: sessionId,
      events_published: eventCount,
      errors: errorCount,
      final_price: lastPrice,
    }, `Adapter shutdown: ${eventCount} events, final $${lastPrice.toFixed(2)}`)
    ws.close()
    setTimeout(() => process.exit(0), 500)
  }

  process.on('SIGINT', shutdown)
  process.on('SIGTERM', shutdown)
}

// ─── Entry ──────────────────────────────────────────────────────────────────

connect()
