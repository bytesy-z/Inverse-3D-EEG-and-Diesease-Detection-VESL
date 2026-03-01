#!/usr/bin/env bash
# =============================================================================
# VESL — PhysDeepSIF Full Stack Start Script (System-Adaptive)
# =============================================================================
#
# Automatically detects the host system's environment and configures paths
# accordingly. Works across different machines without hardcoded paths.
#
# Usage:
#   ./start.sh              # Start both servers
#   ./start.sh --backend    # Start only the backend
#   ./start.sh --frontend   # Start only the frontend
#   ./start.sh --check      # Run checks only (no servers)
#   ./start.sh --kill       # Kill any running servers
#   ./start.sh --info       # Show detected system info
#
# Environment Detection:
#   1. Searches for conda env 'deepsif' in common locations
#   2. Falls back to system Python if conda not found
#   3. Auto-detects Node.js via nvm, volta, or system PATH
#   4. Validates critical files before startup
#
# =============================================================================
set -euo pipefail

# ---- Auto-Detect Project Root ----
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configuration (will be populated by detection) ----
PYTHON=""
NODE=""
NPM=""
BACKEND_PORT=8000
FRONTEND_PORT=3000
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

# Test dataset is OPTIONAL for demo purposes
REQUIRE_TEST_DATASET="${REQUIRE_TEST_DATASET:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'
DIM='\033[2m'

# Global PIDs for cleanup
BACKEND_PID=""
FRONTEND_PID=""

# Detected system info (populated by detect_system)
DETECTED_OS=""
DETECTED_CONDA_BASE=""
DETECTED_DEEPSIF_ENV=""
DETECTED_NVM_DIR=""
DETECTED_NODE_VERSION=""

# ---- Helper Functions ----

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[  OK]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[FAIL]${NC} $*"; }
log_header()  { echo -e "\n${BOLD}${CYAN}=== $* ===${NC}\n"; }
log_dim()     { echo -e "${DIM}       $*${NC}"; }

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

# =============================================================================
# SYSTEM DETECTION — Finds conda, Python, Node.js across different systems
# =============================================================================

detect_os() {
    case "$(uname -s)" in
        Linux*)   DETECTED_OS="Linux" ;;
        Darwin*)  DETECTED_OS="macOS" ;;
        MINGW*|MSYS*|CYGWIN*) DETECTED_OS="Windows" ;;
        *)        DETECTED_OS="Unknown" ;;
    esac
    log_info "Operating System: $DETECTED_OS ($(uname -m))"
}

detect_conda() {
    log_info "Searching for conda installation with 'deepsif' environment..."
    
    # Common conda base locations to search
    local conda_locations=(
        "$HOME/anaconda3"
        "$HOME/miniconda3"
        "$HOME/miniforge3"
        "$HOME/mambaforge"
        "/opt/conda"
        "/opt/anaconda3"
        "/opt/miniconda3"
        "/usr/local/anaconda3"
        "/usr/local/miniconda3"
        "$HOME/.conda"
    )
    
    # Also check CONDA_PREFIX if already in a conda env
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        # Extract base from current env
        local base="${CONDA_PREFIX%/envs/*}"
        if [[ "$base" != "$CONDA_PREFIX" ]]; then
            conda_locations=("$base" "${conda_locations[@]}")
        else
            conda_locations=("$CONDA_PREFIX" "${conda_locations[@]}")
        fi
    fi
    
    # Check conda command directly
    if command -v conda &>/dev/null; then
        local conda_info
        conda_info=$(conda info --base 2>/dev/null || true)
        if [[ -n "$conda_info" && -d "$conda_info" ]]; then
            conda_locations=("$conda_info" "${conda_locations[@]}")
        fi
    fi
    
    # PRIORITY: First try to find conda that has the 'deepsif' environment
    for loc in "${conda_locations[@]}"; do
        if [[ -d "$loc/envs/deepsif/bin" ]]; then
            DETECTED_CONDA_BASE="$loc"
            log_success "Found conda with deepsif env at: $DETECTED_CONDA_BASE"
            return 0
        fi
    done
    
    # Fallback: Find any valid conda base (even without deepsif)
    for loc in "${conda_locations[@]}"; do
        if [[ -d "$loc" && ( -f "$loc/bin/conda" || -f "$loc/condabin/conda" ) ]]; then
            DETECTED_CONDA_BASE="$loc"
            log_success "Found conda at: $DETECTED_CONDA_BASE"
            return 0
        fi
    done
    
    log_warn "Conda installation not found"
    return 1
}

detect_deepsif_env() {
    log_info "Searching for 'deepsif' conda environment..."
    
    if [[ -z "$DETECTED_CONDA_BASE" ]]; then
        log_warn "Cannot search for deepsif env - conda base not detected"
        return 1
    fi
    
    # Standard env location
    local env_path="$DETECTED_CONDA_BASE/envs/deepsif"
    
    # Also check if deepsif is the base env (unlikely but possible)
    if [[ -f "$DETECTED_CONDA_BASE/envs/deepsif/bin/python" ]]; then
        DETECTED_DEEPSIF_ENV="$env_path"
        PYTHON="$env_path/bin/python"
        log_success "Found deepsif env: $DETECTED_DEEPSIF_ENV"
        return 0
    fi
    
    # Search in envs directory
    if [[ -d "$DETECTED_CONDA_BASE/envs" ]]; then
        for env_dir in "$DETECTED_CONDA_BASE/envs"/*; do
            if [[ -d "$env_dir" && "$(basename "$env_dir")" == "deepsif" ]]; then
                if [[ -f "$env_dir/bin/python" ]]; then
                    DETECTED_DEEPSIF_ENV="$env_dir"
                    PYTHON="$env_dir/bin/python"
                    log_success "Found deepsif env: $DETECTED_DEEPSIF_ENV"
                    return 0
                fi
            fi
        done
    fi
    
    log_error "Conda environment 'deepsif' not found"
    log_dim "Create it with: conda create -n deepsif python=3.9"
    log_dim "Then install packages: conda activate deepsif && pip install -r requirements.txt"
    return 1
}

detect_node() {
    log_info "Searching for Node.js..."
    
    # Method 1: Check nvm
    local nvm_locations=(
        "$HOME/.nvm"
        "${NVM_DIR:-}"
        "/usr/local/nvm"
    )
    
    for nvm_dir in "${nvm_locations[@]}"; do
        if [[ -n "$nvm_dir" && -s "$nvm_dir/nvm.sh" ]]; then
            DETECTED_NVM_DIR="$nvm_dir"
            export NVM_DIR="$nvm_dir"
            # shellcheck source=/dev/null
            source "$nvm_dir/nvm.sh" 2>/dev/null || true
            break
        fi
    done
    
    # Method 2: Check volta
    if [[ -d "$HOME/.volta/bin" ]]; then
        export PATH="$HOME/.volta/bin:$PATH"
    fi
    
    # Method 3: Check common system locations
    local node_locations=(
        "/usr/local/bin/node"
        "/usr/bin/node"
        "$HOME/.local/bin/node"
    )
    
    # Try to find node
    if command -v node &>/dev/null; then
        NODE="$(command -v node)"
        NPM="$(command -v npm)"
        DETECTED_NODE_VERSION="$(node --version 2>/dev/null || echo 'unknown')"
        log_success "Found Node.js: $DETECTED_NODE_VERSION at $NODE"
        return 0
    fi
    
    # Check explicit paths
    for node_path in "${node_locations[@]}"; do
        if [[ -x "$node_path" ]]; then
            NODE="$node_path"
            NPM="${node_path%/node}/npm"
            DETECTED_NODE_VERSION="$($NODE --version 2>/dev/null || echo 'unknown')"
            log_success "Found Node.js: $DETECTED_NODE_VERSION at $NODE"
            return 0
        fi
    done
    
    log_error "Node.js not found"
    log_dim "Install with nvm: curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash"
    log_dim "Then: nvm install 20"
    return 1
}

detect_system() {
    log_header "System Detection"
    
    detect_os
    
    if detect_conda; then
        detect_deepsif_env || true
    fi
    
    # Fallback: try system Python with required packages
    if [[ -z "$PYTHON" ]]; then
        log_info "Checking for system Python with required packages..."
        if command -v python3 &>/dev/null; then
            local sys_python
            sys_python="$(command -v python3)"
            # Check if it has torch
            if "$sys_python" -c "import torch" 2>/dev/null; then
                PYTHON="$sys_python"
                log_success "Using system Python: $PYTHON"
                log_warn "Note: Using system Python instead of conda env 'deepsif'"
            fi
        fi
    fi
    
    detect_node || true
    
    # Summary
    echo ""
    log_info "Detection Summary:"
    log_dim "  Project Root:  $PROJECT_ROOT"
    log_dim "  Conda Base:    ${DETECTED_CONDA_BASE:-'not found'}"
    log_dim "  DeepSIF Env:   ${DETECTED_DEEPSIF_ENV:-'not found'}"
    log_dim "  Python:        ${PYTHON:-'not found'}"
    log_dim "  Node.js:       ${NODE:-'not found'} ${DETECTED_NODE_VERSION:-''}"
    log_dim "  NVM Dir:       ${DETECTED_NVM_DIR:-'not found'}"
}

# =============================================================================
# DEPENDENCY CHECKS — Validate packages, versions, and files
# =============================================================================

check_python() {
    log_info "Checking Python environment..."
    
    if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
        log_error "Python executable not found or not set"
        log_dim "Run: ./start.sh --info to see detection results"
        return 1
    fi
    
    local py_version
    py_version=$("$PYTHON" --version 2>&1)
    log_success "Python: $py_version"
    
    # Check critical Python packages with version info
    log_info "Checking required Python packages..."
    
    local missing=()
    local pkg_list="torch fastapi uvicorn h5py numpy scipy mne"
    
    for pkg in $pkg_list; do
        local installed_ver
        installed_ver=$("$PYTHON" -c "
import sys
try:
    import $pkg
    ver = getattr($pkg, '__version__', 'unknown')
    print(ver)
except ImportError:
    print('MISSING')
" 2>/dev/null)
        
        if [[ "$installed_ver" == "MISSING" ]]; then
            missing+=("$pkg")
        else
            log_success "  $pkg: $installed_ver"
        fi
    done
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing Python packages: ${missing[*]}"
        log_dim "Install with: $PYTHON -m pip install ${missing[*]}"
        return 1
    fi
    
    # Check GPU
    local gpu_info
    gpu_info=$("$PYTHON" -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA {torch.version.cuda} - {torch.cuda.get_device_name(0)}')
else:
    print('CPU only')
" 2>/dev/null)
    
    if [[ "$gpu_info" == "CPU only" ]]; then
        log_warn "GPU: $gpu_info (inference will be slower)"
    else
        log_success "GPU: $gpu_info"
    fi
}

check_model() {
    log_info "Checking model and data files..."
    
    # Critical files that MUST exist
    local critical_files=(
        "$PROJECT_ROOT/outputs/models/checkpoint_best.pt:Model checkpoint"
        "$PROJECT_ROOT/outputs/models/normalization_stats.json:Normalization stats"
        "$PROJECT_ROOT/data/leadfield_19x76.npy:Leadfield matrix"
        "$PROJECT_ROOT/data/connectivity_76.npy:Connectivity matrix"
        "$PROJECT_ROOT/data/region_labels_76.json:Region labels"
        "$PROJECT_ROOT/data/region_centers_76.npy:Region centers"
    )
    
    # Optional files (nice to have but not required for demo)
    local optional_files=(
        "$PROJECT_ROOT/data/synthetic3/test_dataset.h5:Test dataset"
        "$PROJECT_ROOT/data/synthetic3/train_dataset.h5:Training dataset"
        "$PROJECT_ROOT/data/synthetic3/val_dataset.h5:Validation dataset"
    )
    
    local all_ok=true
    local critical_missing=()
    
    # Check critical files
    for entry in "${critical_files[@]}"; do
        local filepath="${entry%%:*}"
        local desc="${entry##*:}"
        
        if [[ -f "$filepath" ]]; then
            local size
            size=$(du -h "$filepath" 2>/dev/null | cut -f1)
            log_success "Found: $(basename "$filepath") ($size) - $desc"
        else
            log_error "Missing: $(basename "$filepath") - $desc"
            critical_missing+=("$filepath")
            all_ok=false
        fi
    done
    
    # Check optional files (warn only)
    echo ""
    log_info "Checking optional files..."
    for entry in "${optional_files[@]}"; do
        local filepath="${entry%%:*}"
        local desc="${entry##*:}"
        
        if [[ -f "$filepath" ]]; then
            local size
            size=$(du -h "$filepath" 2>/dev/null | cut -f1)
            log_success "Found: $(basename "$filepath") ($size) - $desc"
        else
            if [[ "$REQUIRE_TEST_DATASET" == "true" ]]; then
                log_error "Missing: $(basename "$filepath") - $desc"
                all_ok=false
            else
                log_warn "Optional: $(basename "$filepath") - $desc (not required for demo)"
            fi
        fi
    done
    
    if [[ ${#critical_missing[@]} -gt 0 ]]; then
        echo ""
        log_error "Critical files missing! The application cannot start."
        log_dim "Make sure you have:"
        for f in "${critical_missing[@]}"; do
            log_dim "  - $f"
        done
        return 1
    fi
    
    $all_ok || return 1
}

check_node() {
    log_info "Checking Node.js environment..."
    
    # Re-source nvm if available (in case environment changed)
    if [[ -n "$DETECTED_NVM_DIR" && -s "$DETECTED_NVM_DIR/nvm.sh" ]]; then
        export NVM_DIR="$DETECTED_NVM_DIR"
        # shellcheck source=/dev/null
        source "$NVM_DIR/nvm.sh" 2>/dev/null || true
    fi
    
    if [[ -z "$NODE" ]] || ! command -v node &>/dev/null; then
        log_error "Node.js not found"
        log_dim "Install with nvm: nvm install 20"
        return 1
    fi
    
    local node_ver
    node_ver=$(node --version 2>/dev/null)
    local node_major="${node_ver#v}"
    node_major="${node_major%%.*}"
    
    if [[ "$node_major" -lt 18 ]]; then
        log_warn "Node.js $node_ver detected, but v18+ recommended"
    else
        log_success "Node.js: $node_ver"
    fi
    
    # Check frontend directory exists
    if [[ ! -d "$FRONTEND_DIR" ]]; then
        log_error "Frontend directory not found: $FRONTEND_DIR"
        return 1
    fi
    
    # Check frontend dependencies
    if [[ ! -d "$FRONTEND_DIR/node_modules/next" ]]; then
        log_warn "node_modules not installed. Installing..."
        cd "$FRONTEND_DIR"
        npm install --legacy-peer-deps 2>&1 | tail -5
        cd "$PROJECT_ROOT"
    fi
    log_success "node_modules: installed"
    
    # Check SWC binary integrity (Linux only)
    if [[ "$DETECTED_OS" == "Linux" ]]; then
        check_swc_binary
    fi
    
    # Verify Next.js loads
    cd "$FRONTEND_DIR"
    if node -e "require('next')" 2>/dev/null; then
        log_success "Next.js: loads OK"
    else
        log_error "Next.js failed to load"
        log_dim "Try: rm -rf frontend/node_modules && cd frontend && npm install --legacy-peer-deps"
        return 1
    fi
    cd "$PROJECT_ROOT"
}

check_swc_binary() {
    local arch
    arch="$(uname -m)"
    local swc_variant=""
    
    case "$arch" in
        x86_64)  swc_variant="linux-x64-gnu" ;;
        aarch64) swc_variant="linux-arm64-gnu" ;;
        *)       log_warn "Unknown architecture: $arch, skipping SWC check"; return 0 ;;
    esac
    
    local swc_bin="$FRONTEND_DIR/node_modules/@next/swc-$swc_variant/next-swc.$swc_variant.node"
    
    if [[ -f "$swc_bin" ]]; then
        local swc_size
        # Handle both Linux and macOS stat syntax
        swc_size=$(stat -c%s "$swc_bin" 2>/dev/null || stat -f%z "$swc_bin" 2>/dev/null || echo 0)
        
        if [[ "$swc_size" -lt 100000000 ]]; then
            log_warn "SWC binary appears truncated (${swc_size} bytes, expected ~143MB)"
            log_info "Attempting to fix SWC binary..."
            fix_swc_binary "$swc_variant"
        else
            log_success "SWC binary: OK ($(( swc_size / 1048576 ))MB)"
        fi
    else
        log_warn "SWC binary not found for $swc_variant, will install..."
        fix_swc_binary "$swc_variant"
    fi
    
    # Verify SWC loads
    cd "$FRONTEND_DIR"
    if node -e "require('@next/swc-$swc_variant')" 2>/dev/null; then
        log_success "SWC runtime: loads OK"
    else
        log_warn "SWC binary check failed (may still work)"
    fi
    cd "$PROJECT_ROOT"
}

fix_swc_binary() {
    local swc_variant="${1:-linux-x64-gnu}"
    local swc_dir="$FRONTEND_DIR/node_modules/@next/swc-$swc_variant"
    
    # Get Next.js version
    local next_ver
    next_ver=$(node -e "console.log(require('$FRONTEND_DIR/node_modules/next/package.json').version)" 2>/dev/null || echo "15.5.4")
    
    log_info "Downloading @next/swc-$swc_variant@$next_ver..."
    local tmp_tgz="/tmp/swc-$swc_variant-$next_ver.tgz"
    local tmp_dir="/tmp/swc-extract-$$"
    
    curl -sL -o "$tmp_tgz" "https://registry.npmjs.org/@next/swc-$swc_variant/-/swc-$swc_variant-$next_ver.tgz" || {
        log_error "Failed to download SWC binary"
        return 1
    }
    
    mkdir -p "$tmp_dir"
    tar xzf "$tmp_tgz" -C "$tmp_dir"
    
    local src_bin="$tmp_dir/package/next-swc.$swc_variant.node"
    if [[ -f "$src_bin" ]]; then
        mkdir -p "$swc_dir"
        cp "$src_bin" "$swc_dir/"
        [[ -f "$tmp_dir/package/package.json" ]] && cp "$tmp_dir/package/package.json" "$swc_dir/"
        local new_size
        new_size=$(stat -c%s "$swc_dir/next-swc.$swc_variant.node" 2>/dev/null || echo 0)
        log_success "SWC binary replaced: $(( new_size / 1048576 ))MB"
    else
        log_error "Failed to extract SWC binary from tarball"
    fi
    
    rm -rf "$tmp_tgz" "$tmp_dir"
}

check_ports() {
    log_info "Checking port availability..."
    
    local port_issues=false
    for port in $BACKEND_PORT $FRONTEND_PORT; do
        local pid=""
        
        # Try ss (Linux)
        if command -v ss &>/dev/null; then
            pid=$(ss -tlnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' || true)
        fi
        
        # Fallback to lsof (macOS/Linux)
        if [[ -z "$pid" ]] && command -v lsof &>/dev/null; then
            pid=$(lsof -ti ":$port" 2>/dev/null || true)
        fi
        
        # Fallback to netstat
        if [[ -z "$pid" ]] && command -v netstat &>/dev/null; then
            pid=$(netstat -tlnp 2>/dev/null | grep ":$port " | awk '{print $NF}' | cut -d'/' -f1 || true)
        fi
        
        if [[ -n "$pid" ]]; then
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
        
        # Re-check ports
        for port in $BACKEND_PORT $FRONTEND_PORT; do
            local pid=""
            if command -v ss &>/dev/null; then
                pid=$(ss -tlnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' || true)
            elif command -v lsof &>/dev/null; then
                pid=$(lsof -ti ":$port" 2>/dev/null || true)
            fi
            
            if [[ -n "$pid" ]]; then
                log_error "Port $port still in use (PID $pid) after kill attempt"
                log_dim "Try manually: kill $pid"
                return 1
            fi
        done
        log_success "All ports now available"
    fi
}

kill_servers() {
    log_header "Killing Running Servers"
    local killed=false
    
    # Kill backend
    local pids
    pids=$(pgrep -f "uvicorn.*server:app" 2>/dev/null || true)
    [[ -z "$pids" ]] && pids=$(pgrep -f "server\.py" 2>/dev/null || true)
    [[ -z "$pids" ]] && pids=$(pgrep -f "fastapi" 2>/dev/null || true)
    
    if [[ -n "$pids" ]]; then
        echo "$pids" | xargs kill 2>/dev/null || true
        log_success "Killed backend server(s): $pids"
        killed=true
    fi
    
    # Kill frontend
    pids=$(pgrep -f "next dev" 2>/dev/null || true)
    [[ -z "$pids" ]] && pids=$(pgrep -f "next-server" 2>/dev/null || true)
    
    if [[ -n "$pids" ]]; then
        echo "$pids" | xargs kill 2>/dev/null || true
        log_success "Killed frontend server(s): $pids"
        killed=true
    fi
    
    # Clean up orphaned next-router-worker
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

# =============================================================================
# SERVER STARTERS
# =============================================================================

start_backend() {
    log_header "Starting Backend (FastAPI on port $BACKEND_PORT)"
    
    if [[ -z "$PYTHON" ]]; then
        log_error "Python not configured"
        return 1
    fi
    
    if [[ ! -f "$BACKEND_DIR/server.py" ]]; then
        log_error "Backend server.py not found at $BACKEND_DIR"
        return 1
    fi
    
    cd "$BACKEND_DIR"
    "$PYTHON" server.py &
    BACKEND_PID=$!
    cd "$PROJECT_ROOT"
    
    log_info "Waiting for backend to load model..."
    local attempts=0
    local max_attempts=60
    
    while [[ $attempts -lt $max_attempts ]]; do
        if curl -s "http://localhost:$BACKEND_PORT/api/health" 2>/dev/null | grep -q '"ok"'; then
            log_success "Backend ready (PID $BACKEND_PID)"
            return 0
        fi
        sleep 1
        attempts=$((attempts + 1))
        
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
    
    # Re-source nvm
    if [[ -n "$DETECTED_NVM_DIR" && -s "$DETECTED_NVM_DIR/nvm.sh" ]]; then
        export NVM_DIR="$DETECTED_NVM_DIR"
        # shellcheck source=/dev/null
        source "$NVM_DIR/nvm.sh" 2>/dev/null || true
    fi
    
    if [[ ! -d "$FRONTEND_DIR" ]]; then
        log_error "Frontend directory not found: $FRONTEND_DIR"
        return 1
    fi
    
    cd "$FRONTEND_DIR"
    node node_modules/.bin/next dev -p "$FRONTEND_PORT" &
    FRONTEND_PID=$!
    cd "$PROJECT_ROOT"
    
    log_info "Waiting for frontend to compile..."
    local attempts=0
    local max_attempts=45
    
    while [[ $attempts -lt $max_attempts ]]; do
        local code
        code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$FRONTEND_PORT/" 2>/dev/null || echo "000")
        
        if [[ "$code" == "200" || "$code" == "307" ]]; then
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

# =============================================================================
# SMOKE TESTS (Optional - only if test data available)
# =============================================================================

run_smoke_test() {
    log_header "Smoke Test"
    
    # Test 1: Backend health
    log_info "Testing backend health..."
    local health
    health=$(curl -s "http://localhost:$BACKEND_PORT/api/health" 2>/dev/null || echo '{}')
    
    if echo "$health" | grep -q '"ok"'; then
        log_success "Backend health: OK"
    else
        log_error "Backend health check failed"
        return 1
    fi
    
    # Test 2: Frontend pages
    log_info "Testing frontend pages..."
    for page in "/" "/analysis"; do
        local code
        code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$FRONTEND_PORT$page" 2>/dev/null || echo "000")
        
        if [[ "$code" == "200" || "$code" == "307" ]]; then
            log_success "GET $page -> $code"
        else
            log_warn "GET $page -> $code (may be normal for redirects)"
        fi
    done
    
    # Test 3: API proxy (only if test data exists)
    local test_data="$PROJECT_ROOT/data/synthetic3/test_dataset.h5"
    if [[ -f "$test_data" ]]; then
        log_info "Testing inference pipeline (requires test data)..."
        local samples
        samples=$(curl -s "http://localhost:$FRONTEND_PORT/api/test-samples?mode=epileptogenic&limit=3" 2>/dev/null || echo '{}')
        
        if echo "$samples" | "$PYTHON" -c "import sys,json; d=json.load(sys.stdin); assert d.get('count',0)>0" 2>/dev/null; then
            log_success "API proxy /api/test-samples: OK"
            
            # Full inference test
            log_info "Running inference on sample_idx=10..."
            local result
            result=$(curl -s -X POST "http://localhost:$FRONTEND_PORT/api/physdeepsif" \
                -F "sample_idx=10" -F "threshold_percentile=87.5" 2>/dev/null || echo '{}')
            
            local success
            success=$(echo "$result" | "$PYTHON" -c "import sys,json; d=json.load(sys.stdin); print(d.get('success',''))" 2>/dev/null || echo "")
            
            if [[ "$success" == "True" ]]; then
                local max_ei max_region
                max_ei=$(echo "$result" | "$PYTHON" -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['epileptogenicity']['max_score']:.3f}\")" 2>/dev/null || echo "?")
                max_region=$(echo "$result" | "$PYTHON" -c "import sys,json; d=json.load(sys.stdin); print(d['epileptogenicity']['max_score_region'])" 2>/dev/null || echo "?")
                log_success "Inference: max_EI=$max_ei ($max_region)"
            else
                log_warn "Inference test inconclusive"
            fi
        else
            log_warn "Test samples not available (test data may be missing)"
        fi
    else
        log_info "Skipping inference test (test_dataset.h5 not found)"
        log_dim "This is normal for demo mode. Upload your own EEG files to test."
    fi
    
    echo ""
}

# =============================================================================
# UI HELPERS
# =============================================================================

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
    echo -e "  ${GREEN}Dashboard:${NC}  http://localhost:$FRONTEND_PORT/analysis"
    echo -e "  ${GREEN}Landing:${NC}    http://localhost:$FRONTEND_PORT"
    echo -e "  ${GREEN}API:${NC}        http://localhost:$BACKEND_PORT/api/health"
    echo ""
    echo -e "  ${YELLOW}Press Ctrl+C to stop all servers${NC}"
    echo ""
}

print_system_info() {
    log_header "System Information"
    
    echo -e "${BOLD}  Host System:${NC}"
    echo -e "    OS:           $DETECTED_OS ($(uname -m))"
    echo -e "    Hostname:     $(hostname)"
    echo -e "    User:         $(whoami)"
    echo ""
    
    echo -e "${BOLD}  Project:${NC}"
    echo -e "    Root:         $PROJECT_ROOT"
    echo -e "    Backend:      $BACKEND_DIR"
    echo -e "    Frontend:     $FRONTEND_DIR"
    echo ""
    
    echo -e "${BOLD}  Environment:${NC}"
    echo -e "    Conda Base:   ${DETECTED_CONDA_BASE:-'not found'}"
    echo -e "    DeepSIF Env:  ${DETECTED_DEEPSIF_ENV:-'not found'}"
    echo -e "    Python:       ${PYTHON:-'not found'}"
    if [[ -n "$PYTHON" && -x "$PYTHON" ]]; then
        echo -e "    Python Ver:   $("$PYTHON" --version 2>&1)"
    fi
    echo ""
    
    echo -e "${BOLD}  Node.js:${NC}"
    echo -e "    NVM Dir:      ${DETECTED_NVM_DIR:-'not found'}"
    echo -e "    Node:         ${NODE:-'not found'}"
    echo -e "    Node Ver:     ${DETECTED_NODE_VERSION:-'unknown'}"
    echo ""
    
    echo -e "${BOLD}  Ports:${NC}"
    echo -e "    Backend:      $BACKEND_PORT"
    echo -e "    Frontend:     $FRONTEND_PORT"
    echo ""
    
    echo -e "${BOLD}  Options:${NC}"
    echo -e "    REQUIRE_TEST_DATASET: $REQUIRE_TEST_DATASET"
    echo ""
}

# =============================================================================
# MAIN
# =============================================================================

main() {
    local mode="${1:-all}"
    
    cd "$PROJECT_ROOT"
    print_banner
    
    case "$mode" in
        --kill|-k)
            kill_servers
            exit 0
            ;;
        --info|-i)
            detect_system
            print_system_info
            exit 0
            ;;
        --check|-c)
            detect_system
            log_header "Dependency Checks"
            check_python
            check_model
            check_node
            check_ports
            log_header "All Checks Passed"
            exit 0
            ;;
        --backend|-b)
            detect_system
            log_header "Dependency Checks"
            check_python
            check_model
            check_ports
            start_backend
            print_urls
            wait "$BACKEND_PID"
            ;;
        --frontend|-f)
            detect_system
            log_header "Dependency Checks"
            check_node
            start_frontend
            print_urls
            wait "$FRONTEND_PID"
            ;;
        all|--all|-a|"")
            detect_system
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
            wait -n "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
            ;;
        --help|-h)
            echo "Usage: $0 [OPTION]"
            echo ""
            echo "Options:"
            echo "  --all, -a       Start both backend and frontend (default)"
            echo "  --backend, -b   Start only the backend server"
            echo "  --frontend, -f  Start only the frontend server"
            echo "  --check, -c     Run dependency checks only"
            echo "  --kill, -k      Kill any running servers"
            echo "  --info, -i      Show detected system information"
            echo "  --help, -h      Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  REQUIRE_TEST_DATASET=true   Require test_dataset.h5 (default: false)"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $mode"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
}

main "$@"
