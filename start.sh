#!/usr/bin/env bash
# =============================================================================
# VESL — PhysDeepSIF Full Stack Start Script
# =============================================================================
#
# Starts both the Python FastAPI backend (port 8000) and the Next.js frontend
# (port 3000) in a single terminal.  Handles dependency checks, SWC binary
# validation, and graceful shutdown.
#
# Usage:
#   ./start.sh              # Start both servers
#   ./start.sh --backend    # Start only the backend
#   ./start.sh --frontend   # Start only the frontend
#   ./start.sh --check      # Run checks only (no servers)
#   ./start.sh --kill       # Kill any running servers
#
# Requirements:
#   - conda env 'deepsif' at /home/tukl/anaconda3/envs/deepsif
#   - nvm + Node.js 20+ at ~/.nvm
#   - Frontend deps installed in frontend/node_modules
#   - Trained model checkpoint at outputs/models/checkpoint_best.pt
#
# =============================================================================
set -euo pipefail

# ---- Configuration ----
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/tukl/anaconda3/envs/deepsif/bin/python"
BACKEND_PORT=8000
FRONTEND_PORT=3000
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Global PIDs for cleanup
BACKEND_PID=""
FRONTEND_PID=""

# ---- Helper Functions ----

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[  OK]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[FAIL]${NC} $*"; }
log_header()  { echo -e "\n${BOLD}${CYAN}=== $* ===${NC}\n"; }

# Gracefully shut down both servers on SIGINT / SIGTERM / EXIT
cleanup() {
    echo ""
    log_header "Shutting Down"
    if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        log_info "Stopping backend (PID $BACKEND_PID)..."
        kill "$BACKEND_PID" 2>/dev/null
        wait "$BACKEND_PID" 2>/dev/null || true
        log_success "Backend stopped"
    fi
    if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        log_info "Stopping frontend (PID $FRONTEND_PID)..."
        kill "$FRONTEND_PID" 2>/dev/null
        wait "$FRONTEND_PID" 2>/dev/null || true
        log_success "Frontend stopped"
    fi
    # Also kill any orphan next-router-worker processes
    pkill -f "next-router-worker" 2>/dev/null || true
    log_info "Goodbye!"
}

trap cleanup EXIT INT TERM

# ---- Dependency Checks ----

check_python() {
    log_info "Checking Python environment..."

    if [[ ! -x "$PYTHON" ]]; then
        log_warn "Configured Python not found at $PYTHON"
        log_info "Searching for a compatible Python/conda env on this machine..."
        # Search common conda env locations for a python binary that can import required packages
        candidates=(
          /home/*/anaconda3/envs/*/bin/python
          /home/*/miniconda3/envs/*/bin/python
          /home/*/miniconda/*/envs/*/bin/python
          /home/*/miniconda/envs/*/bin/python
          /opt/conda/envs/*/bin/python
          /usr/bin/python3
          /usr/local/bin/python3
          /home/zik/miniconda3/envs/*/bin/python
        )
        found=""
        for ppat in "${candidates[@]}"; do
            for cand in $ppat; do
                if [[ -x "$cand" ]]; then
                    # Quick import test
                    if "$cand" - <<'PY' 2>/dev/null
import sys
try:
    import torch, fastapi, uvicorn, h5py, numpy, scipy
    print('OK')
except Exception:
    sys.exit(1)
PY
                    then
                        found="$cand"
                        break 2
                    fi
                fi
            done
        done
        if [[ -n "$found" ]]; then
            PYTHON="$found"
            export PYTHON
            log_success "Found compatible Python: $PYTHON"
        else
            log_error "No compatible Python/conda env found. Install required packages in a conda env and update start.sh PYTHON variable."
            return 1
        fi
    fi
    local py_version
    py_version=$("$PYTHON" --version 2>&1)
    log_success "Python: $py_version"

    # Check critical Python packages
    local missing=()
    for pkg in torch fastapi uvicorn h5py numpy scipy; do
        if ! "$PYTHON" -c "import $pkg" 2>/dev/null; then
            missing+=("$pkg")
        fi
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing Python packages: ${missing[*]}"
        log_info "Install with: $PYTHON -m pip install ${missing[*]}"
        return 1
    fi
    log_success "Python packages: torch, fastapi, uvicorn, h5py, numpy, scipy"

    # Check GPU
    local gpu_status
    gpu_status=$("$PYTHON" -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')" 2>/dev/null)
    if [[ "$gpu_status" == "CUDA" ]]; then
        log_success "GPU: CUDA available"
    else
        log_warn "GPU: CUDA not available, using CPU (inference will be slower)"
    fi
}

check_model() {
    log_info "Checking model files..."

    local checkpoint="$PROJECT_ROOT/outputs/models/checkpoint_best.pt"
    local norm_stats="$PROJECT_ROOT/outputs/models/normalization_stats.json"
    local leadfield="$PROJECT_ROOT/data/leadfield_19x76.npy"
    local connectivity="$PROJECT_ROOT/data/connectivity_76.npy"
    local labels="$PROJECT_ROOT/data/region_labels_76.json"
    local centers="$PROJECT_ROOT/data/region_centers_76.npy"
    local test_data="$PROJECT_ROOT/data/synthetic3/test_dataset.h5"

    local all_ok=true
    for f in "$checkpoint" "$norm_stats" "$leadfield" "$connectivity" "$labels" "$centers"; do
        if [[ -f "$f" ]]; then
            log_success "Found: $(basename "$f")"
        else
            log_error "Missing: $f"
            all_ok=false
        fi
    done

    if [[ -f "$test_data" ]]; then
        log_success "Found: $(basename "$test_data")"
    else
        log_warn "Missing optional test dataset: $test_data"
        log_warn "sample_idx-based testing will be unavailable; file uploads still work"
    fi

    $all_ok || return 1
}

check_node() {
    log_info "Checking Node.js environment..."

    # Load nvm
    export NVM_DIR="$HOME/.nvm"
    if [[ -s "$NVM_DIR/nvm.sh" ]]; then
        # shellcheck source=/dev/null
        source "$NVM_DIR/nvm.sh"
    fi

    if ! command -v node &>/dev/null; then
        log_error "Node.js not found. Install with nvm: nvm install 20"
        return 1
    fi

    local node_ver
    node_ver=$(node --version)
    log_success "Node.js: $node_ver"

    # Check frontend dependencies
    if [[ ! -d "$FRONTEND_DIR/node_modules/next" ]]; then
        log_warn "node_modules not installed. Installing..."
        cd "$FRONTEND_DIR"
        npm install --legacy-peer-deps 2>&1 | tail -3
        cd "$PROJECT_ROOT"
    fi
    log_success "node_modules: installed"

    # Check SWC binary integrity
    local swc_bin="$FRONTEND_DIR/node_modules/@next/swc-linux-x64-gnu/next-swc.linux-x64-gnu.node"
    if [[ -f "$swc_bin" ]]; then
        local swc_size
        swc_size=$(stat -c%s "$swc_bin" 2>/dev/null || stat -f%z "$swc_bin" 2>/dev/null)
        if [[ "$swc_size" -lt 100000000 ]]; then
            log_warn "SWC binary appears truncated (${swc_size} bytes, expected ~143MB)"
            log_info "Attempting to fix SWC binary..."
            fix_swc_binary
        else
            log_success "SWC binary: OK ($(( swc_size / 1048576 ))MB)"
        fi
    else
        log_warn "SWC binary not found, will install..."
        fix_swc_binary
    fi

    # Verify SWC actually loads
    cd "$FRONTEND_DIR"
    if node -e "require('@next/swc-linux-x64-gnu')" 2>/dev/null; then
        log_success "SWC runtime: loads OK"
    else
        log_error "SWC binary crashes on load!"
        log_info "Try: rm -rf frontend/node_modules && npm install --legacy-peer-deps"
        return 1
    fi
    cd "$PROJECT_ROOT"
}

fix_swc_binary() {
    # Downloads the correct SWC binary from npm registry and manually places it
    local swc_dir="$FRONTEND_DIR/node_modules/@next/swc-linux-x64-gnu"
    local next_ver
    next_ver=$(node -e "console.log(require('$FRONTEND_DIR/node_modules/next/package.json').version)" 2>/dev/null || echo "15.5.4")

    log_info "Downloading @next/swc-linux-x64-gnu@$next_ver..."
    local tmp_tgz="/tmp/swc-gnu-$next_ver.tgz"
    local tmp_dir="/tmp/swc-extract-$$"

    curl -sL -o "$tmp_tgz" "https://registry.npmjs.org/@next/swc-linux-x64-gnu/-/swc-linux-x64-gnu-$next_ver.tgz"
    mkdir -p "$tmp_dir"
    tar xzf "$tmp_tgz" -C "$tmp_dir"

    local src_bin="$tmp_dir/package/next-swc.linux-x64-gnu.node"
    if [[ -f "$src_bin" ]]; then
        mkdir -p "$swc_dir"
        cp "$src_bin" "$swc_dir/next-swc.linux-x64-gnu.node"
        # Also copy package.json if present
        [[ -f "$tmp_dir/package/package.json" ]] && cp "$tmp_dir/package/package.json" "$swc_dir/package.json"
        local new_size
        new_size=$(stat -c%s "$swc_dir/next-swc.linux-x64-gnu.node" 2>/dev/null || echo 0)
        log_success "SWC binary replaced: $(( new_size / 1048576 ))MB"
    else
        log_error "Failed to extract SWC binary from tarball"
    fi

    rm -rf "$tmp_tgz" "$tmp_dir"
}

_port_in_use() {
    local port=$1
    ss -tln 2>/dev/null | grep -qP ":$port "
}

check_ports() {
    log_info "Checking port availability..."

    local port_issues=false
    for port in $BACKEND_PORT $FRONTEND_PORT; do
        if _port_in_use "$port"; then
            local pid
            pid=$(ss -tlnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' || echo "(Docker/proxy)")
            log_warn "Port $port already in use (PID $pid)"
            port_issues=true
        else
            log_success "Port $port: available"
        fi
    done

    if $port_issues; then
        log_info "Killing existing processes to free ports..."
        kill_servers
        sleep 2
        # Verify ports are now free
        local still_busy=()
        for port in $BACKEND_PORT $FRONTEND_PORT; do
            if _port_in_use "$port"; then
                still_busy+=("$port")
            fi
        done

        # If frontend port still busy, try fallback ports
        if [[ " ${still_busy[*]} " =~ " $FRONTEND_PORT " ]]; then
            local old_port=$FRONTEND_PORT
            for fallback in 3003 3004 3005 3006 3007; do
                if ! _port_in_use "$fallback"; then
                    FRONTEND_PORT=$fallback
                    log_warn "Port $old_port is held by a Docker container. Using fallback port $FRONTEND_PORT."
                    still_busy=(${still_busy[@]/$old_port})
                    break
                fi
            done
        fi

        for port in "${still_busy[@]}"; do
            log_error "Port $port still in use after cleanup — cannot start."
            log_info "Stop the service on port $port or update start.sh FRONTEND_PORT."
        done
        if [[ ${#still_busy[@]} -gt 0 ]]; then
            return 1
        fi
        log_success "All ports now available"
    fi
}

kill_servers() {
    log_header "Killing Running Servers"
    local killed=false

    # Kill backend — match uvicorn or server.py or python running on backend port
    local pids
    pids=$(pgrep -f "uvicorn.*server:app" 2>/dev/null || true)
    if [[ -z "$pids" ]]; then
        pids=$(pgrep -f "server\.py" 2>/dev/null || true)
    fi
    # Also check by port
    if [[ -z "$pids" ]]; then
        pids=$(ss -tlnp 2>/dev/null | grep ":$BACKEND_PORT " | grep -oP 'pid=\K[0-9]+' || true)
    fi
    if [[ -n "$pids" ]]; then
        echo "$pids" | xargs kill 2>/dev/null || true
        log_success "Killed backend server(s): $pids"
        killed=true
    fi

    # Kill frontend — match next dev, next-server, or process on frontend port
    pids=$(pgrep -f "next dev" 2>/dev/null || true)
    if [[ -z "$pids" ]]; then
        pids=$(pgrep -f "next-server" 2>/dev/null || true)
    fi
    if [[ -z "$pids" ]]; then
        pids=$(ss -tlnp 2>/dev/null | grep ":$FRONTEND_PORT " | grep -oP 'pid=\K[0-9]+' || true)
    fi
    if [[ -n "$pids" ]]; then
        echo "$pids" | xargs kill 2>/dev/null || true
        log_success "Killed frontend server(s): $pids"
        killed=true
    fi

    # Clean up any orphaned next-router-worker processes
    pids=$(pgrep -f "next-router-worker" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        echo "$pids" | xargs kill 2>/dev/null || true
        killed=true
    fi

    if ! $killed; then
        log_info "No running servers found"
    fi

    sleep 1
}

# ---- Server Starters ----

start_backend() {
    log_header "Starting Backend (FastAPI on port $BACKEND_PORT)"

    mkdir -p "$PROJECT_ROOT/logs"
    cd "$BACKEND_DIR"
    "$PYTHON" server.py > "$PROJECT_ROOT/logs/backend.log" 2>&1 &
    BACKEND_PID=$!
    cd "$PROJECT_ROOT"

    # Wait for backend to be ready (model loading takes a few seconds)
    log_info "Waiting for backend to load model..."
    local attempts=0
    local max_attempts=60  # 60 seconds max
    while [[ $attempts -lt $max_attempts ]]; do
        if curl -s "http://localhost:$BACKEND_PORT/api/health" | grep -q '"ok"' 2>/dev/null; then
            log_success "Backend ready (PID $BACKEND_PID)"
            return 0
        fi
        sleep 1
        attempts=$((attempts + 1))

        # Check if process died
        if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
            log_error "Backend process died during startup"
            return 1
        fi
    done

    log_error "Backend did not become ready within ${max_attempts}s"
    return 1
}

start_frontend() {
    log_header "Starting Frontend (Next.js on port $FRONTEND_PORT)"

    # Load nvm for this subshell
    export NVM_DIR="$HOME/.nvm"
    [[ -s "$NVM_DIR/nvm.sh" ]] && source "$NVM_DIR/nvm.sh"

    cd "$FRONTEND_DIR"
    node node_modules/.bin/next dev -p "$FRONTEND_PORT" > "$PROJECT_ROOT/logs/frontend.log" 2>&1 &
    FRONTEND_PID=$!
    cd "$PROJECT_ROOT"

    # Wait for frontend to be ready
    log_info "Waiting for frontend to compile..."
    local attempts=0
    local max_attempts=30
    while [[ $attempts -lt $max_attempts ]]; do
        if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$FRONTEND_PORT/" 2>/dev/null | grep -q "200"; then
            log_success "Frontend ready (PID $FRONTEND_PID)"
            return 0
        fi
        sleep 1
        attempts=$((attempts + 1))

        if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
            log_error "Frontend process died during startup"
            return 1
        fi
    done

    log_error "Frontend did not become ready within ${max_attempts}s"
    return 1
}

run_smoke_test() {
    log_header "Smoke Test"

    # Test 1: Backend health
    log_info "Testing backend health..."
    local health
    health=$(curl -s "http://localhost:$BACKEND_PORT/api/health" 2>/dev/null)
    if echo "$health" | grep -q '"ok"'; then
        log_success "Backend health: OK"
    else
        log_error "Backend health check failed"
        return 1
    fi

    # Test 2: Frontend pages
    for page in "/" "/analysis"; do
        local code
        code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$FRONTEND_PORT$page" 2>/dev/null)
        if [[ "$code" == "200" ]]; then
            log_success "GET $page -> $code"
        else
            log_error "GET $page -> $code (expected 200)"
        fi
    done

    # Test 3: API proxy (test-samples)
    local samples
    samples=$(curl -s "http://localhost:$FRONTEND_PORT/api/test-samples?mode=epileptogenic&limit=3" 2>/dev/null)
    if echo "$samples" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['count']>0" 2>/dev/null; then
        log_success "API proxy /api/test-samples: OK"
    else
        log_error "API proxy /api/test-samples: Failed"
    fi

    # Test 4: Full inference pipeline via proxy
    log_info "Testing full inference pipeline (sample_idx=10)..."
    local result
    result=$(curl -s -X POST "http://localhost:$FRONTEND_PORT/api/physdeepsif" \
        -F "sample_idx=10" -F "threshold_percentile=87.5" 2>/dev/null)

    local success
    success=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('success',''))" 2>/dev/null)
    if [[ "$success" == "True" ]]; then
        local max_ei max_region recall
        max_ei=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['epileptogenicity']['max_score']:.3f}\")" 2>/dev/null)
        max_region=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['epileptogenicity']['max_score_region'])" 2>/dev/null)
        recall=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('groundTruth',{}).get('recall'); print(f'{r:.1%}' if r else 'N/A')" 2>/dev/null)
        log_success "Inference: max_EI=$max_ei ($max_region), recall=$recall"
    else
        log_error "Inference pipeline failed: success=$success"
    fi

    echo ""
}

print_banner() {
    echo -e "${BOLD}${CYAN}"
    echo "  ╔═══════════════════════════════════════════════════════╗"
    echo "  ║                                                       ║"
    echo "  ║   VESL — PhysDeepSIF Brain Source Imaging Platform    ║"
    echo "  ║                                                       ║"
    echo "  ╚═══════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_urls() {
    echo ""
    echo -e "${BOLD}  Access the application:${NC}"
    echo ""
    echo -e "  ${GREEN}Frontend:${NC}  http://localhost:$FRONTEND_PORT"
    echo -e "  ${GREEN}Biomarkers:${NC} http://localhost:$FRONTEND_PORT/biomarkers"
    echo -e "  ${GREEN}Backend:${NC}   http://localhost:$BACKEND_PORT/api/health"
    echo ""
    echo -e "  ${YELLOW}Press Ctrl+C to stop all servers${NC}"
    echo ""
}

# ---- Main ----

main() {
    local mode="${1:-all}"

    cd "$PROJECT_ROOT"
    print_banner

    case "$mode" in
        --kill|-k)
            kill_servers
            exit 0
            ;;
        --check|-c)
            log_header "Dependency Checks"
            check_python
            check_model
            check_node
            check_ports
            log_header "All Checks Passed"
            exit 0
            ;;
        --backend|-b)
            log_header "Dependency Checks"
            check_python
            check_model
            check_ports
            start_backend
            print_urls
            wait "$BACKEND_PID"
            ;;
        --frontend|-f)
            log_header "Dependency Checks"
            check_node
            start_frontend
            print_urls
            wait "$FRONTEND_PID"
            ;;
        all|--all|-a|"")
            log_header "Dependency Checks"
            check_python
            check_model
            check_node
            check_ports
            start_backend
            start_frontend
            run_smoke_test
            print_urls
            # Wait for either process to exit
            wait -n "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
            ;;
        *)
            echo "Usage: $0 [--all|--backend|--frontend|--check|--kill]"
            exit 1
            ;;
    esac
}

main "$@"
