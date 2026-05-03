# Phase 3 Analysis Plan

## Current State

Phase 1 (backup/reset to e820166) and Phase 2 (drop-in files) are complete. The codebase is byte-identical to the recovery kit. **Five structural issues remain** that the RECOVERY_PLAN.md does not address:

| # | Issue | Severity |
|---|-------|----------|
| 1 | **BrainVisualization injection race condition** — `useEffect([plotHtml])` creates a cycle where cleanup clears DOM while async CDN scripts still load, causing stale onload callbacks | Critical |
| 2 | **Sticky sidebar grid contradicts the plan** — `page.tsx:476` uses `lg:grid-cols-[40%_60%]` with `lg:sticky`, which the plan explicitly says to avoid | Medium |
| 3 | **CMA-ES blocks entire biomarker dashboard** — when CMA-ES runs, the right column shows only a spinner, hiding detected regions, XAI, and waveform. Spinner must be scoped to concordance card only | Medium |
| 4 | **CMA-ES runtime too long for practical use** — defaults of `maxWindows=5`, `cmaesGens=20`, population=2 means CMA-ES processes 5 windows × 20 generations. Reduce to 2 windows × 2 gens × pop 2 for ~5-10s total | Medium |
| 5 | **Orphaned Next.js API proxy routes** — 6 files under `frontend/app/api/` that the old architecture used but the new page.tsx does not | Low |

**Routing note**: The page.tsx fetches to `/api/analyze` (relative), which works via `next.config.mjs:19` rewrite rule → proxies to backend. This is accidentally correct. The plan specifies `${backendUrl}/api/analyze` (direct). The env var names differ (`PHYSDEEPSIF_BACKEND_URL` server-side vs `NEXT_PUBLIC_PHYSDEEPSIF_BACKEND` client-side) — a potential configuration trap worth hardening.

## Scientific Validity

All computational procedures are sound:
- **Inference**: Global z-score with raw (DC+AC) stats, no per-channel de-meaning — matches PhysDeepSIF training pipeline. EEG retains DC spatial prior.
- **CMA-ES biophysical inversion**: Neural-mass parameter fitting via Covariance Matrix Adaptation — correct black-box optimization for clinical EI validation.
- **Concordance**: Cross-method comparison (network-heuristic EI vs optimization-biophysical EI) — standard clinical biomarker validation methodology.

No scientific errors detected.

## Root Cause: Why Phase 3 Broke Last Time

The debugging output is from the **mega-commit's ~500-line brain-viz** (not the current 394-line version). Three factors cascaded:

### RC1 (PRIMARY): Plotly.animate in cleanup on uninitialized figure
```
[BrainVis] Cleanup: stopping animation
[BrainVis] Plotly.animate called with args: ["{...}","[]","{...}"]
```
The mega-commit's cleanup called `Plotly.animate` to stop playback before clearing the container. When CDN scripts hadn't finished loading, the figure wasn't initialized, causing silent failure. React StrictMode (double-mount in dev) doubled the injection/cancel cycle. The current 394-line version has no `Plotly.animate` call in cleanup — but still has a subtler race (stale CDN onload callbacks executing on containers already cleared by a subsequent injection).

### RC2 (SECONDARY): WebSocket-triggered re-renders during injection
The mega-commit's `useWebSocket` hook received CMA-ES progress each generation. Every `setState` re-rendered the entire analysis page, which re-rendered `BrainVisualization`, triggering cleanup→injection cycles. Every CMA-ES generation = one re-injection cycle. The current architecture replaces WebSocket with `setInterval` polling (2s), which is safer but the underlying injection fragility remains.

### RC3 (TERTIARY): Frame sync dependency on `loading`
The frame sync `useEffect` depends on `[loading, onFrameChange]`. During injection cycling, `loading` toggles, detaching/reattaching Plotly event listeners on potentially stale DOM nodes, producing `Frame sync: skipping (no plotDiv)`.

**Bottom line**: The debugging features and WebSocket were not the root cause alone — they amplified a fundamental race condition in the Plotly HTML injection strategy under React's concurrent rendering. The fix is to harden the injection lifecycle.

## Implementation Plan (6 Edits)

All edits on `frontend/` files. Backend (`server.py`, 2600 lines) is byte-identical to the recovery kit and needs no changes. The backend already supports `max_windows` and `cmaes_generations` params, and `CMAES_POPULATION_SIZE=2` is already the default.

---

### Edit 1: Harden brain-viz injection with AbortController

**File**: `frontend/components/brain-visualization.tsx`, replace the `useEffect` at lines 60-122

**Problem**: The `cancelled` boolean flag pattern doesn't stop in-flight async CDN script loads. After cleanup clears the container, old script `onload` callbacks still fire and mutate stale state.

**Fix**: Use `AbortController`. Gate every async step on `ac.signal.aborted`. On cleanup, call `ac.abort()` first to cancel in-flight operations, then clear DOM.

```typescript
useEffect(() => {
    const container = containerRef.current
    if (!container || !plotHtml) return

    const ac = new AbortController()
    setLoading(true)

    const run = async () => {
      container.innerHTML = ""
      const wrapper = document.createElement("div")
      wrapper.innerHTML = plotHtml
      const scripts = Array.from(wrapper.querySelectorAll("script"))
      scripts.forEach((s) => s.parentNode?.removeChild(s))
      while (wrapper.firstChild) {
        container.appendChild(wrapper.firstChild)
      }

      for (const oldScript of scripts) {
        if (ac.signal.aborted) return
        await new Promise<void>((resolve, reject) => {
          const newScript = document.createElement("script")
          Array.from(oldScript.attributes).forEach((a) =>
            newScript.setAttribute(a.name, a.value)
          )
          const onAbort = () => {
            newScript.remove()
            resolve()
          }
          ac.signal.addEventListener("abort", onAbort, { once: true })
          if (oldScript.src) {
            newScript.onload = () => { resolve() }
            newScript.onerror = () => reject(new Error(`Failed to load: ${oldScript.src}`))
            newScript.src = oldScript.src
          } else {
            newScript.textContent = oldScript.textContent
            container.appendChild(newScript)
            return resolve()
          }
          container.appendChild(newScript)
        })
      }

      if (!ac.signal.aborted) {
        const plotDiv = container.querySelector<PlotlyGraphDiv>(".plotly-graph-div")
        plotDivRef.current = plotDiv ?? null
        resizePlotly(container)
        setLoading(false)
      }
    }

    run().catch(console.error)

    return () => {
      ac.abort()
      plotDivRef.current = null
      container.innerHTML = ""
    }
  }, [plotHtml])
```

---

### Edit 2: Restore simple vertical layout + scope CMA-ES spinner to concordance only

**File**: `frontend/app/analysis/page.tsx`, replace the entire biomarkers view block (lines 476-596)

**Problem**: (a) The sticky sidebar grid `lg:grid-cols-[40%_60%]` with `lg:sticky` is the layout the plan prohibits. (b) When CMA-ES runs, the right column is entirely replaced by a spinner, hiding detected regions and XAI. The spinner must only cover the concordance section so the rest of the dashboard remains visible.

**Fix**: Single-column `space-y-4` layout. The spinner is scoped to an inline loader inside the concordance card, not a full-card replacement. Brain viz, detected regions, XAI, and waveform all render immediately regardless of CMA-ES state.

```tsx
{/* ---- Biomarker Detection View ---- */}
{viewMode === "biomarkers" && bioResult && (
  <div className="space-y-4">
    {/* Window selector */}
    {(esiResult?.eegData?.windows && esiResult.eegData.windows.length > 1) && (
      <div className="flex items-center gap-3 text-sm">
        <span className="text-muted-foreground">Window:</span>
        <div className="flex gap-1">
          {esiResult.eegData.windows.map((_, idx) => (
            <button
              key={idx}
              onClick={() => setClampedWindow(idx)}
              className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                selectedWindow === idx
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-muted-foreground hover:text-foreground"
              }`}
            >
              {idx + 1}
            </button>
          ))}
        </div>
      </div>
    )}

    {/* Brain heatmap — full width, always visible */}
    <BrainVisualization
      plotHtml={bioResult.plotHtml}
      label="Epileptogenicity Map"
      className="h-[640px] w-full"
      currentFrame={selectedWindow}
      onFrameChange={handleBrainFrameChange}
    />

    {/* Section 1: Detected Regions — always visible */}
    <Card>
      <div className="px-4 py-2 border-b">
        <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Epileptogenic Regions</span>
      </div>
      <div className="p-4">
        <DetectedRegions regions={detectedRegions} variant="clinical" />
      </div>
    </Card>

    {/* Section 2: Concordance — shows spinner while CMA-ES runs, badge when complete */}
    <Card>
      <div className="px-4 py-2 border-b">
        <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Biophysical Validation</span>
      </div>
      <div className="p-4">
        {bioResult?.cmaes?.status === "running" ? (
          <div className="flex items-center gap-3 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin text-primary" />
            <span>Running CMA-ES concordance validation (generation {bioResult.cmaes?.generations ?? 0}/{bioResult.cmaes?.max_generations ?? 30})</span>
          </div>
        ) : bioResult?.concordance ? (
          <ConcordanceBadge
            tier={bioResult.concordance.tier}
            overlap={bioResult.concordance.overlap}
            description={bioResult.concordance.tier_description}
            sharedRegions={bioResult.concordance.shared_regions}
          />
        ) : (
          <p className="text-sm text-muted-foreground">Concordance analysis pending</p>
        )}
      </div>
    </Card>

    {/* Section 3: XAI — always visible */}
    {bioResult?.xai && (
      <Card>
        <div className="px-4 py-2 border-b">
          <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Explainability (XAI)</span>
        </div>
        <div className="p-4">
          <XaiPanel
            channelImportance={bioResult.xai.channel_importance ?? []}
            timeImportance={bioResult.xai.time_importance ?? []}
            channelNames={esiResult?.eegData?.channels ?? bioResult?.eegData?.channels ?? DEFAULT_CHANNEL_NAMES}
            topSegments={bioResult.xai.top_segments ?? []}
            eegData={esiResult?.eegData ?? bioResult?.eegData ?? undefined}
            selectedWindow={selectedWindow}
          />
        </div>
      </Card>
    )}

    {/* EEG Waveform — collapsible */}
    {bioResult?.eegData && (
      <details className="group">
        <summary className="text-xs font-medium uppercase tracking-wider text-muted-foreground cursor-pointer select-none hover:text-foreground">
          EEG Waveform {esiResult?.eegData?.windows?.length && esiResult.eegData.windows.length > 1 ? `(${esiResult.eegData.windows.length} windows)` : ""}
        </summary>
        <div className="mt-3">
          <EegWaveformPlot
            eegData={bioResult.eegData}
            selectedWindow={selectedWindow}
            onSelectedWindowChange={setClampedWindow}
            className="h-[400px]"
          />
        </div>
      </details>
    )}
  </div>
)}
```

---

### Edit 3: Set tight CMA-ES defaults (2 windows, 2 gens)

**File**: `frontend/app/analysis/page.tsx`, change state initializers at lines 69-71

**Current**:
```typescript
const [maxWindows, setMaxWindows] = useState<number>(5)
const [cmaesGens, setCmaesGens] = useState<number>(20)
```

**Replace with**:
```typescript
const [maxWindows, setMaxWindows] = useState<number>(2)
const [cmaesGens, setCmaesGens] = useState<number>(2)
```

The backend already has `CMAES_POPULATION_SIZE = 2` (hardcoded). With 2 windows × 2 generations × population 2 = ~5-10 seconds for CMA-ES, making the concordance result appear quickly while still generating meaningful biophysical validation.

---

### Edit 4: Delete orphaned Next.js API proxy routes

```bash
rm -f frontend/app/api/analyze-eeg/route.ts
rm -f frontend/app/api/physdeepsif/route.ts
rm -f frontend/app/api/analyze-mat/route.ts
rm -f frontend/app/api/test-samples/route.ts
```

The new page.tsx uses `/api/analyze` via `next.config.mjs` rewrite (not these deprecated proxy routes). Keep `job-status/route.ts` and `serve-result/[...path]/route.ts`.

---

### Edit 5: Delete unused WebSocket hook

**File**: `frontend/hooks/use-websocket.ts` — delete. This was part of the mega-commit's WebSocket-based architecture. The current page.tsx uses `setInterval` polling for CMA-ES status. Dead code in a hook file is a maintenance trap. If future WebSocket integration is needed, rewrite from scratch with the AbortController patterns from Edit 1.

---

### Edit 6: Align env var naming in next.config.mjs

**File**: `frontend/next.config.mjs:16`

```javascript
// Current (server-side only):
const backendUrl = process.env.PHYSDEEPSIF_BACKEND_URL || 'http://localhost:8000'

// Fix: fall through both possible env var names
const backendUrl = process.env.PHYSDEEPSIF_BACKEND_URL
  || process.env.NEXT_PUBLIC_PHYSDEEPSIF_BACKEND
  || 'http://localhost:8000'
```

---

## Implementation Sequence

Execute in order:

1. **Edit 1** (brain-viz AbortController) — critical fix, affects all views
2. **Edit 2** (restore vertical layout + scoped spinner) — fixes UX for CMA-ES
3. **Edit 3** (tight defaults: 2 windows, 2 gens) — practical speed
4. **Edit 4** (delete orphaned API routes) — cleanup
5. **Edit 5** (delete websocket hook) — cleanup
6. **Edit 6** (env var alignment) — defense

## Verification

After applying:
```bash
./start.sh --backend && ./start.sh --frontend
```

1. Upload an EDF → source localization view renders cleanly (no "Injection cancelled" in console)
2. Switch to biomarkers view → brain heatmap, detected regions, and XAI appear immediately
3. Concordance card shows inline spinner while CMA-ES runs (~5-10s), then displays ConcordanceBadge
4. Frame sync works: brain animation slider advances EEG waveform window
