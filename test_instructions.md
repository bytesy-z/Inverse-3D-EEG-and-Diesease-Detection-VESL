# Verification Steps
1. Open the UI, load a file.
2. Under source localization or biomarker, verify the scale control shows "Scale" and a dropdown with options "5 µV/div", "10 µV/div", ..., "100 µV/div".
3. Check the graph's traces appropriately scale based on the selection.
4. Press the "Play" button on the 3D Brain visualization.
5. Verify it plays cleanly frame-to-frame without stopping or skipping.


Phase 4: Backend smoke test

# Start backend
$PYTHON backend/server.py &
BACKEND_PID=$!
sleep 3

# Health 
curl -sS http://127.0.0.1:8000/api/health | $PYTHON -m json.tool

# Synthetic sample inference #passed
curl -sS -X POST "http://127.0.0.1:8000/api/analyze" \
  -F "sample_idx=10" -F "mode=source_localization" \
  | $PYTHON -c "import sys,json; d=json.load(sys.stdin); print('status:', d.get('status')); print('has plotHtml:', 'plotHtml' in d)"

# EDF upload #passed
curl -sS -X POST "http://127.0.0.1:8000/api/analyze" \
  -F "file=@data/test_demo.edf" -F "mode=source_localization" \
  | $PYTHON -c 'import sys,json; d=json.load(sys.stdin); print("status:", d.get("status")); print("has heatmap:", "plotHtml" in d)'

kill $BACKEND_PID; wait $BACKEND_PID 2>/dev/null

Phase 5: Frontend build #passed

cd frontend
#cd /data1tb/VESL/fyp-2.0/frontend && npm install --legacy-peer-deps
npm run build   # must compile without errors 
npm run lint    # should show no errors

Phase 6: Full E2E

bash scripts/smoke_test.sh   # should exit 0 #passed

Phase 7: Manual UI walkthrough

    bash start.sh (both servers)
    Open http://localhost:3000
    Test: synthetic sample → submit → 3D heatmap renders
    Test: biomarkers tab → concordance badge + XAI panel visible
    Test: EDF upload → results load
    Test: dark mode toggle → UI adapts
    Test: error state (upload .txt) → friendly message, no crash

