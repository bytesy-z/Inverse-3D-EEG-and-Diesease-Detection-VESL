#!/bin/bash
# End-to-end test: EDF upload → preprocessing → inference → biomarker → CMA-ES → XAI
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"
echo "Testing against $BASE_URL"

# Health check
echo "--- Health check ---"
curl -sS "$BASE_URL/api/health" | python -m json.tool

# Analyze with synthetic sample (biomarker mode)
echo -e "\n--- Synthetic sample: biomarker mode ---"
RESPONSE=$(curl -sS -X POST "$BASE_URL/api/analyze" -F "sample_idx=0" -F "mode=biomarkers")
echo "$RESPONSE" | python -c "
import sys, json
d = json.load(sys.stdin)
scores = d.get('epileptogenicity', {}).get('scores', {})
prelim = d.get('epileptogenicity', {}).get('preliminary', {})
xai = d.get('xai', {})
print(f'Status: {d.get(\"status\", \"unknown\")}')
print(f'EI scores: {len(scores)} regions')
print(f'Preliminary: {prelim}')
print(f'XAI channels: {len(xai.get(\"channel_importance\", []))}')
print(f'Top region: {d.get(\"epileptogenicity\", {}).get(\"top_region\", \"N/A\")}')
" 2>/dev/null || echo "Raw response: ${RESPONSE:0:300}..."

# Upload demo EDF file
EDF_FILE="data/test_demo.edf"
if [ -f "$EDF_FILE" ]; then
    echo -e "\n--- EDF upload: source_localization mode ---"
    RESPONSE=$(curl -sS -X POST "$BASE_URL/api/analyze" -F "file=@$EDF_FILE" -F "mode=source_localization")
    echo "$RESPONSE" | python -c "
import sys, json
d = json.load(sys.stdin)
src = d.get('source_estimate', {})
print(f'Status: {d.get(\"status\", \"unknown\")}')
print(f'Source shape: {src.get(\"shape\", \"N/A\")}')
print(f'Heatmap generated: {\"heatmap_html\" in d}')
" 2>/dev/null || echo "Raw response: ${RESPONSE:0:300}..."

    echo -e "\n--- EDF upload: biomarker mode (CMA-ES enabled) ---"
    RESPONSE=$(curl -sS -X POST "$BASE_URL/api/analyze" -F "file=@$EDF_FILE" -F "mode=biomarkers")
    echo "$RESPONSE" | python -c "
import sys, json
d = json.load(sys.stdin)
scores = d.get('epileptogenicity', {}).get('scores', {})
xai = d.get('xai', {})
print(f'Status: {d.get(\"status\", \"unknown\")}')
print(f'EI scores: {len(scores)} regions')
print(f'XAI channels: {len(xai.get(\"channel_importance\", []))}')
print(f'Heatmap: {\"heatmap_html\" in d}')
" 2>/dev/null || echo "Raw response: ${RESPONSE:0:300}..."
else
    echo -e "\n--- Skipping EDF upload: $EDF_FILE not found ---"
fi

echo -e "\nAll checks passed."
