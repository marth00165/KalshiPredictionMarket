#!/usr/bin/env bun
/**
 * Polymarket Pipeline Prometheus Exporter
 *
 * Exposes pipeline metrics as Prometheus text format for Grafana scraping.
 * Queries SurrealDB periodically and serves metrics on /metrics endpoint.
 *
 * Usage:
 *   bun run metrics-exporter.ts                  # Default: port 9191
 *   bun run metrics-exporter.ts --port 9192      # Custom port
 *   bun run metrics-exporter.ts --scrape 15      # Scrape SurrealDB every 15s
 *
 * Metrics exposed:
 *   polymarket_events_total{source}              # Total events by source
 *   polymarket_events_rate{source}               # Events per minute (5m window)
 *   polymarket_severity_total{severity}          # Events by severity
 *   polymarket_signals_total{signal_type}        # Signals by type
 *   polymarket_portfolio_value                   # Current portfolio value
 *   polymarket_portfolio_pnl                     # Realized + unrealized P&L
 *   polymarket_portfolio_positions               # Number of open positions
 *   polymarket_portfolio_win_rate                # Win rate percentage
 *   polymarket_active_markets                    # Markets with trades in last 15m
 *   polymarket_trades_per_minute                 # Overall trade throughput
 *
 * Grafana setup:
 *   1. Add Prometheus data source pointing to this exporter
 *   2. Import dashboard or query polymarket_* metrics
 */

const args = process.argv.slice(2)
function getArg(name: string, fallback: string): string {
  const idx = args.indexOf(`--${name}`)
  return idx >= 0 && args[idx + 1] ? args[idx + 1] : fallback
}

const PORT = parseInt(getArg('port', '9191'))
const SCRAPE_INTERVAL = parseInt(getArg('scrape', '15'))

const SURREAL_URL = 'http://127.0.0.1:8284'
const SURREAL_NS = 'nabi'
const SURREAL_DB = 'substrate'
const SURREAL_AUTH = Buffer.from('root:root').toString('base64')
const PORTFOLIO_STATE = '/home/tryk/.local/state/nabi/portfolio/polymarket.json'


// ─── Cached Metrics ─────────────────────────────────────────────────────────

interface MetricsCache {
  eventsBySource: Map<string, number>
  rateBySource: Map<string, number>
  severityCounts: Map<string, number>
  signalCounts: Map<string, number>
  activeMarkets: number
  tradesPerMinute: number
  portfolioValue: number
  portfolioPnl: number
  portfolioPositions: number
  portfolioWinRate: number
  portfolioCash: number
  portfolioClosedTrades: number
  lastScrape: number
}

let cache: MetricsCache = {
  eventsBySource: new Map(),
  rateBySource: new Map(),
  severityCounts: new Map(),
  signalCounts: new Map(),
  activeMarkets: 0,
  tradesPerMinute: 0,
  portfolioValue: 0,
  portfolioPnl: 0,
  portfolioPositions: 0,
  portfolioWinRate: 0,
  portfolioCash: 0,
  portfolioClosedTrades: 0,
  lastScrape: 0,
}

// ─── SurrealDB Queries ──────────────────────────────────────────────────────

async function query(sql: string): Promise<any[]> {
  try {
    const response = await fetch(`${SURREAL_URL}/sql`, {
      method: 'POST',
      headers: {
        'Authorization': `Basic ${SURREAL_AUTH}`,
        'surreal-ns': SURREAL_NS,
        'surreal-db': SURREAL_DB,
        'Accept': 'application/json',
      },
      body: sql,
    })
    if (!response.ok) return []
    const results: any[] = await response.json()
    return results[0]?.result || []
  } catch {
    return []
  }
}

async function scrapeMetrics(): Promise<void> {
  const now = Date.now()
  const fiveMinAgo = new Date(now - 5 * 60 * 1000).toISOString()
  const fifteenMinAgo = new Date(now - 15 * 60 * 1000).toISOString()

  // Run all queries in parallel — each targets exact source with tight time windows
  const [
    liveCount5m,
    signalsCount5m,
    portfolioCount5m,
    severityCounts,
    signalTypes,
    activeMarkets,
    totalLive,
    totalSignals,
    totalPortfolio,
  ] = await Promise.all([
    // Per-source 5m rate counts
    query(`SELECT count() as cnt FROM plexus_event WHERE source = 'polymarket.live' AND timestamp > '${fiveMinAgo}' GROUP ALL`),
    query(`SELECT count() as cnt FROM plexus_event WHERE source = 'polymarket.signals' AND timestamp > '${fiveMinAgo}' GROUP ALL`),
    query(`SELECT count() as cnt FROM plexus_event WHERE source = 'polymarket.portfolio' AND timestamp > '${fiveMinAgo}' GROUP ALL`),
    // Severity distribution (15m)
    query(`SELECT severity, count() as cnt FROM plexus_event WHERE source = 'polymarket.live' AND timestamp > '${fifteenMinAgo}' GROUP BY severity`),
    // Signal types (15m)
    query(`SELECT event_type, count() as cnt FROM plexus_event WHERE source = 'polymarket.signals' AND timestamp > '${fifteenMinAgo}' GROUP BY event_type`),
    // Active markets (15m) — GROUP BY condition_id, count rows in JS
    query(`SELECT metadata.condition_id, count() as cnt FROM plexus_event WHERE source = 'polymarket.live' AND timestamp > '${fifteenMinAgo}' GROUP BY metadata.condition_id`),
    // Total counts per source (1h window to stay fast — full count is too expensive)
    query(`SELECT count() as cnt FROM plexus_event WHERE source = 'polymarket.live' GROUP ALL`),
    query(`SELECT count() as cnt FROM plexus_event WHERE source = 'polymarket.signals' GROUP ALL`),
    query(`SELECT count() as cnt FROM plexus_event WHERE source = 'polymarket.portfolio' GROUP ALL`),
  ])

  // Total events by source
  cache.eventsBySource.clear()
  if (totalLive[0]?.cnt) cache.eventsBySource.set('polymarket.live', totalLive[0].cnt)
  if (totalSignals[0]?.cnt) cache.eventsBySource.set('polymarket.signals', totalSignals[0].cnt)
  if (totalPortfolio[0]?.cnt) cache.eventsBySource.set('polymarket.portfolio', totalPortfolio[0].cnt)

  // Event rate by source (5m window → per minute)
  cache.rateBySource.clear()
  if (liveCount5m[0]?.cnt) cache.rateBySource.set('polymarket.live', liveCount5m[0].cnt / 5)
  if (signalsCount5m[0]?.cnt) cache.rateBySource.set('polymarket.signals', signalsCount5m[0].cnt / 5)
  if (portfolioCount5m[0]?.cnt) cache.rateBySource.set('polymarket.portfolio', portfolioCount5m[0].cnt / 5)

  // Severity distribution
  cache.severityCounts.clear()
  for (const r of severityCounts) {
    cache.severityCounts.set(r.severity, r.cnt)
  }

  // Signal type distribution
  cache.signalCounts.clear()
  for (const r of signalTypes) {
    const signalType = r.event_type.replace('signal.suggestion.', '')
    cache.signalCounts.set(signalType, r.cnt)
  }

  // Active markets (count of GROUP BY rows = distinct condition IDs)
  cache.activeMarkets = activeMarkets.length

  // Trades per minute (from 5m rate)
  cache.tradesPerMinute = (liveCount5m[0]?.cnt || 0) / 5

  // Portfolio state from file (no SurrealDB query needed)
  try {
    const file = Bun.file(PORTFOLIO_STATE)
    if (await file.exists()) {
      const state = await file.json()
      cache.portfolioValue = state.total_value || 0
      cache.portfolioPnl = state.total_pnl || 0
      cache.portfolioPositions = state.positions?.length || 0
      cache.portfolioCash = state.capital || 0
      cache.portfolioClosedTrades = state.closed_count || 0
      cache.portfolioWinRate = state.closed_count > 0
        ? (state.win_count / state.closed_count) * 100
        : 0
    }
  } catch { /* portfolio not running yet */ }

  cache.lastScrape = now
}

// ─── Prometheus Format ──────────────────────────────────────────────────────

function renderMetrics(): string {
  const lines: string[] = []

  // Helper
  const metric = (name: string, help: string, type: string, value: number, labels?: Record<string, string>) => {
    const labelStr = labels
      ? '{' + Object.entries(labels).map(([k, v]) => `${k}="${v}"`).join(',') + '}'
      : ''
    lines.push(`# HELP ${name} ${help}`)
    lines.push(`# TYPE ${name} ${type}`)
    lines.push(`${name}${labelStr} ${value}`)
  }

  const metricLine = (name: string, value: number, labels?: Record<string, string>) => {
    const labelStr = labels
      ? '{' + Object.entries(labels).map(([k, v]) => `${k}="${v}"`).join(',') + '}'
      : ''
    lines.push(`${name}${labelStr} ${value}`)
  }

  // Events total by source
  lines.push('# HELP polymarket_events_total Total events by source')
  lines.push('# TYPE polymarket_events_total counter')
  for (const [source, count] of cache.eventsBySource) {
    metricLine('polymarket_events_total', count, { source })
  }

  // Event rate by source
  lines.push('# HELP polymarket_events_rate_per_minute Events per minute by source (5m window)')
  lines.push('# TYPE polymarket_events_rate_per_minute gauge')
  for (const [source, rate] of cache.rateBySource) {
    metricLine('polymarket_events_rate_per_minute', parseFloat(rate.toFixed(2)), { source })
  }

  // Severity distribution
  lines.push('# HELP polymarket_severity_count Events by severity in last 15m')
  lines.push('# TYPE polymarket_severity_count gauge')
  for (const [severity, count] of cache.severityCounts) {
    metricLine('polymarket_severity_count', count, { severity })
  }

  // Signal distribution
  lines.push('# HELP polymarket_signals_count Signals by type in last 15m')
  lines.push('# TYPE polymarket_signals_count gauge')
  for (const [signal, count] of cache.signalCounts) {
    metricLine('polymarket_signals_count', count, { signal_type: signal })
  }

  // Active markets
  metric('polymarket_active_markets', 'Markets with trades in last 15m', 'gauge', cache.activeMarkets)

  // Trades per minute
  metric('polymarket_trades_per_minute', 'Trade throughput (5m avg)', 'gauge',
    parseFloat(cache.tradesPerMinute.toFixed(2)))

  // Portfolio metrics
  metric('polymarket_portfolio_value_usd', 'Current portfolio value in USD', 'gauge',
    parseFloat(cache.portfolioValue.toFixed(2)))
  metric('polymarket_portfolio_pnl_usd', 'Total P&L in USD', 'gauge',
    parseFloat(cache.portfolioPnl.toFixed(2)))
  metric('polymarket_portfolio_positions', 'Number of open positions', 'gauge',
    cache.portfolioPositions)
  metric('polymarket_portfolio_cash_usd', 'Available cash in USD', 'gauge',
    parseFloat(cache.portfolioCash.toFixed(2)))
  metric('polymarket_portfolio_closed_trades', 'Total closed trades', 'counter',
    cache.portfolioClosedTrades)
  metric('polymarket_portfolio_win_rate', 'Win rate percentage', 'gauge',
    parseFloat(cache.portfolioWinRate.toFixed(1)))

  // Meta
  metric('polymarket_exporter_last_scrape_timestamp', 'Last scrape timestamp', 'gauge',
    cache.lastScrape / 1000)

  return lines.join('\n') + '\n'
}

// ─── HTTP Server ────────────────────────────────────────────────────────────

const server = Bun.serve({
  port: PORT,
  fetch(req) {
    const url = new URL(req.url)

    if (url.pathname === '/metrics') {
      return new Response(renderMetrics(), {
        headers: { 'Content-Type': 'text/plain; version=0.0.4; charset=utf-8' },
      })
    }

    if (url.pathname === '/health') {
      return new Response(JSON.stringify({
        status: 'ok',
        last_scrape: new Date(cache.lastScrape).toISOString(),
        sources: Object.fromEntries(cache.eventsBySource),
      }), { headers: { 'Content-Type': 'application/json' } })
    }

    // Landing page
    return new Response(
      '<html><body>' +
      '<h1>Polymarket Pipeline Metrics</h1>' +
      '<p><a href="/metrics">/metrics</a> - Prometheus metrics</p>' +
      '<p><a href="/health">/health</a> - Health check</p>' +
      '</body></html>',
      { headers: { 'Content-Type': 'text/html' } }
    )
  },
})

console.log(`[EXPORTER] Prometheus metrics on http://localhost:${PORT}/metrics`)
console.log(`[EXPORTER] Scraping SurrealDB every ${SCRAPE_INTERVAL}s`)

// Initial scrape
await scrapeMetrics()
console.log(`[EXPORTER] Initial scrape complete: ${cache.eventsBySource.size} sources, ${cache.activeMarkets} markets`)

// Periodic scrape
setInterval(async () => {
  try {
    await scrapeMetrics()
  } catch (err: any) {
    console.error(`[SCRAPE ERROR] ${err.message}`)
  }
}, SCRAPE_INTERVAL * 1000)

process.on('SIGINT', () => {
  console.log('\n[SHUTDOWN] Metrics exporter stopped.')
  server.stop()
  process.exit(0)
})
