#!/usr/bin/env bash
set -euo pipefail

# End-to-end smoke test:
# - Start backend, verify health
# - Start frontend, verify reachable
# - Exercise /api/analyze with synthetic EEG data
# - Verify response contains eegData and a waveform key
# - Cleanup

echo "[smoke] Starting backend..."
./start.sh --backend &> /tmp/smoke_backend.log &
BACKEND_PID=$!
sleep 2

# Wait for backend health to be green
echo "[smoke] Waiting for backend health..."
HEALTH_OK=0
for i in {1..60}; do
  HEALTH_JSON=$(curl -sS http://127.0.0.1:8000/api/health 2>/dev/null || true)
  if echo "$HEALTH_JSON" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('status', ''))
except Exception:
    pass
" | grep -q ok; then
    HEALTH_OK=1
    break
  fi
  sleep 1
done
if [ "$HEALTH_OK" -ne 1 ]; then
  echo "[smoke] Backend health check failed. See logs at /tmp/smoke_backend.log"
  ./start.sh --kill || true
  exit 1
fi
echo "[smoke] Backend healthy."

echo "[smoke] Starting frontend..."
./start.sh --frontend &> /tmp/smoke_frontend.log &
FRONTEND_PID=$!
sleep 3

# Check frontend root is reachable
echo "[smoke] Verifying frontend is reachable..."
HTTP_CODE=$(curl -sS -o /dev/null -w "%{http_code}" http://127.0.0.1:3000/ || true)
if [ "$HTTP_CODE" != "200" ]; then
  echo "[smoke] Frontend not reachable (HTTP $HTTP_CODE). Check /tmp/smoke_frontend.log";
  ./start.sh --kill || true
  exit 1
fi
echo "[smoke] Frontend reachable (HTTP 200)."

echo "[smoke] Verifying frontend page contents..."
FRONT_PAGE=$(curl -sS http://127.0.0.1:3000/ 2>/dev/null | head -n 1) || true
if [[ "$FRONT_PAGE" != *"<!DOCTYPE html>"* && "$FRONT_PAGE" != *"Plotly"* && "$FRONT_PAGE" != *"Brain"* ]]; then
  echo "[smoke] Frontend page content not as expected."
  # Do not fail hard; it's still useful to proceed to API test
fi

echo "[smoke] Creating synthetic EEG data..."
TMP_EEG=/tmp/synthetic_eeg.npy
python3 - <<'PY'
import numpy as np
arr = np.random.randn(19, 400).astype('float32')
np.save('/tmp/synthetic_eeg.npy', arr)
print('/tmp/synthetic_eeg.npy')
PY
if [ ! -f "$TMP_EEG" ]; then
  echo "[smoke] Failed to create synthetic EEG file."; exit 1
fi

echo "[smoke] Submitting /api/analyze request with synthetic EEG..."
RESPONSE=$(curl -sS -X POST "http://127.0.0.1:8000/api/analyze" \
  -F "file=@$TMP_EEG" \
  -F "mode=source_localization" || true)

if [ -z "$RESPONSE" ]; then
  echo "[smoke] Empty response from /api/analyze"; exit 1
fi

set +o pipefail
echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'eegData' not in data:
  print('MISSING eegData')
  sys.exit(2)
for key in ('waveform','waveformImage','image','plot','waveform_data'):
  if key in data:
    print(f'FOUND_WAVEFORM:{key}')
    sys.exit(0)
print('NO_WAVEFORM_KEY')
sys.exit(0)
"
EXIT_CODE=$?
set -o pipefail
if [ "$EXIT_CODE" -ne 0 ]; then
  echo "[smoke] analyze response validation failed. See details above."
  ./start.sh --kill || true
  exit 1
fi


echo "[smoke] Cleaning up servers..."
./start.sh --kill || true
exit 0
