#!/usr/bin/env bun
/**
 * Attention Capture — "Send Without Thinking"
 *
 * The simplest possible interface: send a URL or text, the substrate
 * captures it with automatic context (BTC price, timestamp, market state).
 *
 * Usage:
 *   bun run capture.ts "https://x.com/someone/status/123"
 *   bun run capture.ts "Fed rate decision tomorrow — risk off"
 *   bun run capture.ts --reflect <event-id> --assessment reinforced
 *   bun run capture.ts --posture confidence_down --magnitude 0.6
 *
 * The human provides meaning. The substrate provides context.
 */

import { $ } from 'bun'

// ─── Arg parsing ────────────────────────────────────────────────────────────

const args = process.argv.slice(2)
function getArg(name: string, fallback?: string): string | undefined {
  const idx = args.indexOf(`--${name}`)
  return idx >= 0 && args[idx + 1] ? args[idx + 1] : fallback
}
const hasFlag = (name: string) => args.includes(`--${name}`)

// ─── Context snapshot ───────────────────────────────────────────────────────

async function getContextSnapshot(): Promise<Record<string, any>> {
  const snapshot: Record<string, any> = {
    timestamp: new Date().toISOString(),
  }

  // Try to get current BTC price from Coinbase (fast, no auth)
  try {
    const resp = await fetch('https://api.coinbase.com/v2/prices/BTC-USD/spot', {
      signal: AbortSignal.timeout(8000),
    })
    if (resp.ok) {
      const data = await resp.json() as any
      snapshot.btc_price = data.data?.amount || 'unknown'
    }
  } catch {
    snapshot.btc_price = 'unavailable'
  }

  // Market session (crypto is 24/7 but traditional market hours matter for sentiment)
  const hour = new Date().getUTCHours()
  if (hour >= 13 && hour < 14) snapshot.time_of_day = 'pre_market'       // 8-9 ET
  else if (hour >= 14 && hour < 20) snapshot.time_of_day = 'market_open'  // 9-3 ET
  else if (hour >= 20 && hour < 21) snapshot.time_of_day = 'market_close' // 3-4 ET
  else if (new Date().getUTCDay() === 0 || new Date().getUTCDay() === 6) snapshot.time_of_day = 'weekend'
  else snapshot.time_of_day = 'after_hours'

  // Query recent severity counts from substrate (last 1h of market events)
  try {
    const result = await $`curl -s http://localhost:8284/sql \
      -H "Accept: application/json" \
      -H "surreal-ns: nabi" \
      -H "surreal-db: substrate" \
      -H "Authorization: Basic ${Buffer.from('root:root').toString('base64')}" \
      -d "SELECT severity, count() AS cnt FROM plexus_event WHERE source CONTAINS 'market-feed' AND time::now() - timestamp < 1h GROUP BY severity"`.quiet()

    if (result.exitCode === 0) {
      const data = JSON.parse(result.stdout.toString())
      const severities = data[0]?.result || []
      snapshot.recent_severity_counts = {}
      for (const s of severities) {
        snapshot.recent_severity_counts[s.severity] = s.cnt
      }
    }
  } catch {
    // Non-critical — context enrichment is best-effort
  }

  return snapshot
}

// ─── Detect content type ────────────────────────────────────────────────────

function detectContentType(content: string): { type: string; domain?: string } {
  try {
    const url = new URL(content)
    const domain = url.hostname.replace('www.', '')

    if (domain.includes('x.com') || domain.includes('twitter.com')) {
      return { type: 'url', domain: 'x.com' }
    }
    return { type: 'url', domain }
  } catch {
    // Not a URL
    if (content.startsWith('"') || content.startsWith('\u201c')) {
      return { type: 'quote' }
    }
    return { type: 'text', domain: 'manual' }
  }
}

// ─── Publish helper ─────────────────────────────────────────────────────────

async function publish(source: string, eventType: string, severity: string, metadata: Record<string, any>) {
  const metaJson = JSON.stringify(metadata)

  try {
    const result = await $`nabi events publish \
      --source ${source} \
      --event-type ${eventType} \
      --severity ${severity} \
      --metadata ${metaJson}`.quiet()

    if (result.exitCode !== 0) {
      const stderr = result.stderr.toString()
      console.error(`[ERROR] ${stderr.slice(0, 200)}`)
      return null
    }

    // Extract event ID from success message
    const stdout = result.stdout.toString()
    const idMatch = stdout.match(/ID: ([a-f0-9-]+)/)
    return idMatch?.[1] || 'published'
  } catch (err: any) {
    console.error(`[ERROR] ${err.message?.slice(0, 200)}`)
    return null
  }
}

// ─── Mode: Attention Capture ────────────────────────────────────────────────

async function captureAttention(content: string) {
  const { type, domain } = detectContentType(content)
  const context = await getContextSnapshot()

  const metadata = {
    event_type: 'attention.captured',
    content_type: type,
    content,
    source_domain: domain || 'manual',
    context_snapshot: context,
    session_id: crypto.randomUUID(),
    seq: 1,
    description: `Attention: ${content.slice(0, 80)}${content.length > 80 ? '...' : ''}`,
  }

  const eventId = await publish('judgment:attention', 'attention.captured', 'info', metadata)

  if (eventId) {
    console.log(`[CAPTURED] ${type} from ${domain || 'manual'}`)
    console.log(`  BTC: $${context.btc_price} | ${context.time_of_day}`)
    console.log(`  ID: ${eventId}`)
    console.log(`  Content: ${content.slice(0, 100)}${content.length > 100 ? '...' : ''}`)
  }
}

// ─── Mode: Judgment Reflection ──────────────────────────────────────────────

async function reflectOnEvent(eventId: string) {
  const assessment = getArg('assessment', 'ambiguous') as string
  const outcome = getArg('outcome', 'neutral') as string
  const action = getArg('action', 'no_action_correct') as string
  const narrative = getArg('narrative')

  const context = await getContextSnapshot()

  const metadata: Record<string, any> = {
    event_type: 'judgment.reflection',
    references: [eventId],
    assessment,
    outcome_quality: outcome,
    hindsight_action: action,
    price_at_reflection: context.btc_price,
    session_id: crypto.randomUUID(),
    description: `Reflection on ${eventId.slice(0, 8)}: ${assessment} / ${outcome}`,
  }

  if (narrative) metadata.narrative = narrative

  const severity = assessment === 'contradicted' ? 'warning' : 'info'
  const id = await publish('judgment:reflection', 'judgment.reflection', severity, metadata)

  if (id) {
    console.log(`[REFLECTED] ${assessment} / ${outcome}`)
    console.log(`  On: ${eventId}`)
    console.log(`  Hindsight: ${action}`)
    if (narrative) console.log(`  Note: ${narrative}`)
  }
}

// ─── Mode: Posture Shift ────────────────────────────────────────────────────

async function shiftPosture(direction: string) {
  const magnitude = parseFloat(getArg('magnitude', '0.5') || '0.5')
  const previous = getArg('previous', 'unknown') as string
  const next = getArg('next', 'neutral') as string
  const authority = getArg('authority', 'maintain') as string
  const reasoning = getArg('reasoning')

  const severity = (authority === 'halt_new_positions' || authority === 'exit_all')
    ? 'critical'
    : direction.includes('down') || direction === 'narrative_conflict'
      ? 'warning'
      : 'info'

  const metadata: Record<string, any> = {
    event_type: 'posture.shift',
    direction,
    magnitude,
    previous_posture: previous,
    new_posture: next,
    authority_recommendation: authority,
    confidence_score: magnitude,
    session_id: crypto.randomUUID(),
    description: `Posture: ${direction} (${magnitude.toFixed(1)}) → ${authority}`,
  }

  if (reasoning) metadata.reasoning = reasoning

  const id = await publish('judgment:posture', 'posture.shift', severity, metadata)

  if (id) {
    console.log(`[POSTURE] ${direction} (magnitude: ${magnitude.toFixed(1)})`)
    console.log(`  ${previous} → ${next}`)
    console.log(`  Authority: ${authority}`)
  }
}

// ─── Entry ──────────────────────────────────────────────────────────────────

if (hasFlag('reflect')) {
  const eventId = getArg('reflect')
  if (!eventId) {
    console.error('Usage: capture.ts --reflect <event-id> --assessment <reinforced|contradicted|...>')
    process.exit(1)
  }
  await reflectOnEvent(eventId)
} else if (hasFlag('posture')) {
  const direction = getArg('posture')
  if (!direction) {
    console.error('Usage: capture.ts --posture <confidence_up|confidence_down|...> --magnitude 0.5')
    process.exit(1)
  }
  await shiftPosture(direction)
} else if (hasFlag('help')) {
  console.log(`
Attention Capture — Judgment Substrate CLI

CAPTURE (default):
  bun run capture.ts "https://x.com/someone/status/123"
  bun run capture.ts "Fed decision tomorrow"

REFLECT:
  bun run capture.ts --reflect <event-id> --assessment reinforced --outcome acceptable
  Assessments: reinforced, contradicted, ambiguous, irrelevant, premature, late
  Outcomes: acceptable, unacceptable, neutral, exceeded
  Actions: increase_exposure, decrease_exposure, hold, exit, no_action_correct

POSTURE:
  bun run capture.ts --posture confidence_down --magnitude 0.7 --authority decrease_exposure_limit
  Directions: confidence_up, confidence_down, uncertainty_up, uncertainty_down, narrative_conflict, regime_shift
  Authority: increase_exposure_limit, decrease_exposure_limit, maintain, halt_new_positions, exit_all
`)
} else if (args.length > 0 && !args[0].startsWith('--')) {
  await captureAttention(args[0])
} else {
  console.error('Usage: bun run capture.ts "<url or text>"')
  console.error('       bun run capture.ts --help')
  process.exit(1)
}
