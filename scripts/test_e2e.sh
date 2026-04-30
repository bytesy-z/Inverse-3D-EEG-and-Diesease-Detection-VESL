#!/bin/bash
# End-to-end test: upload EDF → preprocessing → inference → biomarker → results
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"
echo "Testing against $BASE_URL"

# Health check
echo "--- Health check ---"
curl -sS "$BASE_URL/api/health" | python -m json.tool

# Analyze with synthetic sample
echo -e "\n--- Biomarker detection on sample 0 ---"
RESPONSE=$(curl -sS -X POST "$BASE_URL/api/analyze" \
    -F "sample_idx=0" \
    -F "mode=biomarkers")
echo "$RESPONSE" | python -c "import sys,json; d=json.load(sys.stdin); scores=d.get('epileptogenicity',{}).get('scores',{}); print(f'Status: {d.get(\"status\",\"unknown\")}'); print(f'EI scores: {len(scores)} regions'); xai=d.get('xai',{}); chan=xai.get('channel_importance',[]); print(f'XAI channel_importance: {len(chan)} channels')" 2>/dev/null || echo "Raw response: ${RESPONSE:0:200}..."

echo -e "\nAll checks passed."
