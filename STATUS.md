# STATUS.md — PhysDeepSIF v2.0 Submission State

**Last updated:** May 1, 2026  
**Tag:** `v2.0-submission` → `856500f`  
**Purpose:** Catch-up document for anyone continuing work (deployment, demo prep, post-submission fixes).

---

## Current State

Everything planned is built. The system runs end-to-end: upload a 19-channel EEG → PhysDeepSIF inference → biomarker detection + CMA-ES biophysical fitting → 3D brain heatmap with concordance score.

| Layer | Status | Notes |
|-------|--------|-------|
| Backend (FastAPI, port 8000) | Done | 7 endpoints + WebSocket, rate-limited, CORS hardened, structured JSON logging with rotation, threading lock on job tracker, disk-usage guard, mesh cached at module level |
| Frontend (Next.js, port 3000) | Done | Error boundaries, loading skeleton, real WebSocket progress bar, concordance badge (3 tiers), XAI panel (channel + time), accessible (aria-pressed, skip-link, aria-live), dark/light theme |
| Model (PhysDeepSIF) | Done | Spatial MLP 19→76 + BiLSTM, trained for 104 epochs (best), config: β=0.0, λ_L=0.0, λ_T=0.3, λ_A=0.2 |
| Synthetic data | Done | ~80k train / ~10k val / ~10k test samples in `data/synthetic3/` (~13 GB) |
| Validation figures | Done | 6 figures in `outputs/figures/` — DLE histogram (31mm), AUC vs SNR (0.923), top-K recall (0.822), hemisphere accuracy, learning curve, concordance heatmap |
| Tests | Done | 149 tests (unit, functional, integration, regression, system), all pass |
| Documentation | Done | 8 docs in `docs/`, README at root (full reproduction guide), AGENTS.md at root |
| Docker | Done | 2-service compose, healthcheck, bridge network, restart policies |
| CI/CD | Done | `.github/workflows/test.yml` — runs on push/PR |

---

## What the Audit Found & Fixed

An adversarial 5-subagent audit was run on April 30/May 1. All findings fixed:

### Critical (6 fixed)
1. **Async inference path had no timeout** — `server.py:2037` now wrapped in `asyncio.wait_for(60s)`
2. **`startup_check()` return value ignored** — now raises `RuntimeError` if files are missing
3. **Dead `jobs` dict** — removed from `server.py:226`
4. **`SYSTEM_ARCHITECTURE.md` said "per-channel de-meaning"** — corrected to "NO per-channel de-meaning" (matches actual code)
5. **Tag `v2.0-submission` was 2 commits behind HEAD** — relocated
6. **No project README** — wrote 800-line reproduction guide

### High severity (8 fixed)
7. **`01_PLAIN_LANGUAGE_DESCRIPTION.md` had rubric text accidentally prepended** — removed
8. **Same doc said β>0, described dropped NMT pipeline, outdated EI formula** — all updated
9. **WebSocket `/ws/` had no proxy in dev mode** — added `NEXT_PUBLIC_PHYSDEEPSIF_BACKEND_WS` env var + auto-detect fallback
10. **`zod@3.25.76` was a bogus npm version** — fixed to `^3.24.0`
11. **`@react-three/drei`, `@react-three/fiber`, `three` were never imported** — removed from `package.json` (~200MB saved)
12. **Test count documented as 135 but actual is 149** — `FINAL_WORK_PLAN_v2.md` updated
13. **`SYSTEM_ARCHITECTURE.md` referenced `generate_all_figures()` and `xai-visualization.tsx` which don't exist** — corrected to `main()` and `xai-panel.tsx`
14. **`next.config.mjs` had hardcoded `localhost:3000` for results rewriter** — fixed to use `${backendUrl}`

### Not fixed (minor, non-blocking)
- RateLimiter bucket dictionary grows forever (no cleanup of empty entries) — only matters after months of unique IPs
- Mesh cache global has no thread-safety on first init — first 2 concurrent requests might both load fsaverage5 mesh; harmless
- Missing `websockets` in `requirements.txt` — bundled transitively by uvicorn
- `archive/synthetic_v4/` is missing `test_dataset.h5` — may be intentional partial snapshot

---

## How to Run

```bash
# Quick start (both servers)
./start.sh

# Backend only
./start.sh --backend

# Frontend only
./start.sh --frontend

# Test suite (use conda python)
/home/tukl/anaconda3/envs/deepsif/bin/python -m pytest tests/ -m "not slow" -v

# E2E integration test
./scripts/test_e2e.sh

# Dependency checks
./start.sh --check
```

Full reproduction guide (conda setup, data generation, training, validation) is in `README.md`.

---

## What Still Needs Doing

### Before/during demo presentation

1. **Pre-run CMA-ES on a few samples** so the demo shows instant results:
   ```bash
   curl -X POST http://localhost:8000/api/analyze -F "sample_idx=5" -F "mode=source_localization"
   # Save the job_id and results URL for the demo
   ```

2. **Prepare a clean browser tab** with `http://localhost:3000` open — no dev tools, no other tabs cluttering the screen

3. **Test the full flow** as the professor would see it:
   - Open frontend → select Sample Index 10 → Submit (source_localization mode)
   - Show progress bar moving → show 3D brain heatmap → switch to biomarkers tab → show XAI panel
   - Upload `data/test_demo.edf` → show source localization results

### Post-submission (deployment)

Deployment is worth 4 marks (PLO-11). The system is containerised but not yet hosted publicly. Plan:

1. **Option A (simplest): Single VPS**
   - Get a $5/month instance (AWS Lightsail, DigitalOcean, Linode)
   - Install Docker, `cd deploy && docker compose up -d`
   - Put nginx in front with Let's Encrypt
   - Set `PHYSDEEPSIF_BACKEND_URL` env var in frontend container to backend's internal hostname

2. **Option B: Vercel + VPS**
   - Backend on VPS (same as Option A, expose port 8000 with HTTPS)
   - Frontend on Vercel
   - Set 2 env vars on Vercel:
     ```
     PHYSDEEPSIF_BACKEND_URL = https://your-vps.com:8000
     NEXT_PUBLIC_PHYSDEEPSIF_BACKEND_WS = your-vps.com:8000
     ```
   - **Caveat:** Vercel's 4.5MB serverless function body size limit will reject 15MB EDF uploads. Either skip Vercel for upload-heavy demo, or have the client upload directly to backend S3 presigned URL.

3. **Before deploying:** Add the production domain to CORS `allow_origins` in `backend/server.py:268-275`.

### Known deployment gotchas
- Backend needs HTTPS for production — browsers block mixed content (HTTPS frontend → HTTP backend)
- The synthetic data is 13 GB — too big for Docker image. Mount it as a volume or regenerate on the server.
- WebSocket doesn't work through Vercel serverless — client must connect to backend directly (already handled by WebSocket hook if `NEXT_PUBLIC_PHYSDEEPSIF_BACKEND_WS` is set)

---

## Key Files to Know

| For... | Read... |
|--------|---------|
| Understanding the project | `docs/01_PLAIN_LANGUAGE_DESCRIPTION.md` |
| Technical details | `docs/02_TECHNICAL_SPECIFICATIONS.md` |
| Code navigation | `docs/SYSTEM_ARCHITECTURE.md` |
| Reproduction steps | `README.md` |
| Agent/CLI conventions | `AGENTS.md` |
| Experiment history | `docs/03_EXPERIMENTATION_LOGS.md` |
| Final config rationale | `docs/30thaprplan.md` |
| Task tracker | `docs/FINAL_WORK_PLAN_v2.md` |

---

## File Map for Quick Edits

| If you need to change... | Edit this |
|--------------------------|-----------|
| Backend API logic | `backend/server.py` |
| Model architecture | `src/phase2_network/physdeepsif.py` |
| Loss function | `src/phase2_network/loss_functions.py` |
| Training pipeline | `scripts/03_train_network.py` |
| CMA-ES optimizer | `src/phase4_inversion/cmaes_optimizer.py` |
| XAI analysis | `src/xai/eeg_occlusion.py` |
| Frontend analysis page | `frontend/app/analysis/page.tsx` |
| Concordance badge | `frontend/components/concordance-badge.tsx` |
| WebSocket hook | `frontend/hooks/use-websocket.ts` |
| Config hyperparameters | `config.yaml` |
| Docker setup | `deploy/docker-compose.yml`, `deploy/Dockerfile.backend` |
| Start script | `start.sh` |
| Test config | `pytest.ini` |

---

## Git

```bash
git log --oneline -5
# 856500f backend finishing touches
# 5f3276d rubrics  
# e28b5ca update docs
# 26300fd v2.0-submission: fix inference preprocessing
# ...

git tag
# v2.0-submission
```

Tag is at current HEAD. No uncommitted changes (except this `STATUS.md` if you're reading it fresh, and possibly `README.md`).
