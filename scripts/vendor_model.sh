#!/bin/bash
# Copy the trained Delhi model + feature config from pollu_sop into the repo so
# kashi serves a self-contained artifact. Re-run after retraining.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${POLLU_SOP_ROOT:-/Users/shlokjain/pollu_sop}/models/delhi_all_features"
DST="$ROOT/kashi/data/model"

mkdir -p "$DST"
cp "$SRC/lightgbm_d3_final.joblib" "$DST/"
cp "$SRC/feature_config.json" "$DST/"
echo "vendored model -> $DST"
ls -la "$DST"
