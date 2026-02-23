#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/nba_tomorrow_injuries.sh
#   ./scripts/nba_tomorrow_injuries.sh 2026-02-24
#
# Required:
#   SPORTRADAR_API_KEY (or SPORTSRADAR_API_KEY alias)
# Optional:
#   SPORTRADAR_ACCESS_LEVEL=trial|production
#   SPORTRADAR_LANG=en

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load .env only if key not already exported in shell.
if [[ -z "${SPORTRADAR_API_KEY:-}" && -z "${SPORTSRADAR_API_KEY:-}" && -f "${REPO_ROOT}/.env" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi

API_KEY="${SPORTRADAR_API_KEY:-${SPORTSRADAR_API_KEY:-}}"
ACCESS_LEVEL="${SPORTRADAR_ACCESS_LEVEL:-trial}"
LANG_CODE="${SPORTRADAR_LANG:-en}"

if [[ -z "${API_KEY}" ]]; then
  echo "ERROR: SPORTRADAR_API_KEY is not set." >&2
  echo "Set it in your shell or .env, then rerun." >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required (brew install jq)." >&2
  exit 1
fi

DATE_INPUT="${1:-}"

# Default to today's date (local timezone)
if [[ -z "${DATE_INPUT}" ]]; then
  TARGET_DATE="$(date +%Y-%m-%d)"
else
  TARGET_DATE="${DATE_INPUT}"
fi

if [[ ! "${TARGET_DATE}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "ERROR: date must be YYYY-MM-DD (got '${TARGET_DATE}')." >&2
  exit 1
fi

YEAR="${TARGET_DATE:0:4}"
MONTH="${TARGET_DATE:5:2}"
DAY="${TARGET_DATE:8:2}"

BASE_URL="https://api.sportradar.com/nba/${ACCESS_LEVEL}/v8/${LANG_CODE}"
SCHEDULE_URL="${BASE_URL}/games/${YEAR}/${MONTH}/${DAY}/schedule.json"
INJURIES_URL="${BASE_URL}/league/injuries.json"

echo "Date: ${TARGET_DATE}"
echo "Fetching games and league injuries from SportsRadar (${ACCESS_LEVEL}/${LANG_CODE})..."

SCHEDULE_JSON="$(curl -fsSL "${SCHEDULE_URL}" \
  --header "accept: application/json" \
  --header "x-api-key: ${API_KEY}")"
INJURIES_JSON="$(curl -fsSL "${INJURIES_URL}" \
  --header "accept: application/json" \
  --header "x-api-key: ${API_KEY}")"

TEAM_REFS="$(jq -r '
  [(.games // [])[] |
    (.home.reference // .home.alias // .home.abbr // empty),
    (.away.reference // .away.alias // .away.abbr // empty)
  ] | map(select(length > 0)) | unique[]' <<< "${SCHEDULE_JSON}")"

if [[ -z "${TEAM_REFS}" ]]; then
  echo "No NBA games found for ${TARGET_DATE}."
  exit 0
fi

TEAM_JSON="$(printf '%s\n' "${TEAM_REFS}" | jq -R . | jq -s .)"

echo
echo "NBA injury report for teams playing on ${TARGET_DATE}"
echo "====================================================="

jq -r --argjson teams "${TEAM_JSON}" '
  [(.teams // [])[] as $t
    | select(($teams | index($t.reference // "")) != null)
    | ($t.players // [])[] as $p
    | ($p.injuries // [])[] as $i
    | {
        team: ($t.reference // $t.alias // $t.name // "UNK"),
        player: ($p.full_name // (($p.first_name // "") + " " + ($p.last_name // "")) | gsub("^\\s+|\\s+$"; "")),
        status: ($i.status // "UNKNOWN"),
        desc: ($i.desc // ""),
        comment: ($i.comment // "")
      }
  ] as $rows
  | if ($rows | length) == 0 then
      "No listed injuries for teams in games on this date."
    else
      ($rows[]
       | "\(.team) | \(.player) | \(.status) | \(.desc)"
         + (if .comment != "" then " | \(.comment)" else "" end))
    end
' <<< "${INJURIES_JSON}"
