#!/usr/bin/env bash
set -e

# Minimal pipeline runner per user instructions.
# Prompts (or accepts via flags) for: start-date, end-date, openaq-token, ee-project, bbox, city
# Then runs the three commands exactly as requested.

usage() {
  echo "Usage: $0 [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--openaq-token TOKEN] [--ee-project PROJECT] [--bbox W,S,E,N] [--city CITY]"
  exit 1
}

# Read args (optional); if missing, prompt interactively
while [[ $# -gt 0 ]]; do
  case $1 in
    --start-date)
      START_DATE="$2"; shift 2;;
    --end-date)
      END_DATE="$2"; shift 2;;
    --openaq-token)
      OPENAQ_TOKEN="$2"; shift 2;;
    --ee-project)
      EE_PROJECT="$2"; shift 2;;
    --bbox)
      BBOX="$2"; shift 2;;
    --city)
      CITY="$2"; shift 2;;
    -h|--help)
      usage;;
    *)
      usage;;
  esac
done

START_DATE="${START_DATE:-$INPUT}"
END_DATE="${END_DATE:-$INPUT}"
OPENAQ_TOKEN="${OPENAQ_TOKEN:-$INPUT}"
EE_PROJECT="${EE_PROJECT:-$INPUT}"
BBOX="${BBOX:-$INPUT}"
CITY="${CITY:-$INPUT}"

# Validate required
if [[ -z "$START_DATE" || -z "$END_DATE" || -z "$OPENAQ_TOKEN" || -z "$EE_PROJECT" || -z "$BBOX" || -z "$CITY" ]]; then
  echo "Error: all arguments are required." >&2
  exit 1
fi

# Run the three commands in sequence (use openaq-token value for --api-token)
python -m patliputra.pipeline run -v \
  --city "$CITY" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --api-token "$OPENAQ_TOKEN" \
  --ee-project "$EE_PROJECT" \
  --hourly

python -m nalanda.target_cli \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --openaq-token "$OPENAQ_TOKEN" \
  --bbox "$BBOX"

python -m nalanda.cli assemble --start-date "$START_DATE" --end-date "$END_DATE" --parallel --workers 8

exit $?
