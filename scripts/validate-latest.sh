#!/usr/bin/env bash
# validate-latest.sh — one-shot gate: test LNS backend against latest ggml, llama.cpp, and xlnscpp HEAD.
#
# Usage:
#   MODEL_SMOLLM=/path/SmolLM2-135M-Instruct-Q4_K_M.gguf \
#   MODEL_LLAMA32=/path/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
#   ./scripts/validate-latest.sh
#
# Exits 0 only if unit test + both inference runs succeed.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
XLNSCPP_DIR="$REPO_ROOT/xlnscpp"
XLNSCPP_BASE_COMMIT="$(cat "$REPO_ROOT/patches/xlnscpp-base-commit.txt")"
BUILD_DIR="$REPO_ROOT/build/validate-latest"
LOG_FILE="$BUILD_DIR/validate-latest.log"

MODEL_SMOLLM="${MODEL_SMOLLM:-$REPO_ROOT/SmolLM2-135M-Instruct-Q4_K_M.gguf}"
MODEL_LLAMA32="${MODEL_LLAMA32:-$REPO_ROOT/Llama-3.2-1B-Instruct-Q4_K_M.gguf}"

MODEL_SMOLLM="$(cd "$(dirname "$MODEL_SMOLLM")" && pwd)/$(basename "$MODEL_SMOLLM")"
MODEL_LLAMA32="$(cd "$(dirname "$MODEL_LLAMA32")" && pwd)/$(basename "$MODEL_LLAMA32")"

for m in "$MODEL_SMOLLM" "$MODEL_LLAMA32"; do
    if [ ! -f "$m" ]; then
        echo "ERROR: Model not found: $m" >&2
        exit 1
    fi
done

if command -v nproc &>/dev/null; then
    JOBS=$(nproc)
else
    JOBS=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
fi

mkdir -p "$BUILD_DIR"
: > "$LOG_FILE"

log() { echo "$@" | tee -a "$LOG_FILE"; }

copy_backend() {
    local dest_src="$1"
    local dest_inc="$2"
    mkdir -p "$dest_src"
    cp "$REPO_ROOT/lns-backend/src/ggml-lns.cpp" "$dest_src/"
    cp "$REPO_ROOT/lns-backend/src/lns-ops.cpp"  "$dest_src/"
    cp "$REPO_ROOT/lns-backend/src/lns-ops.h"    "$dest_src/"
    cp "$REPO_ROOT/lns-backend/src/CMakeLists.txt" "$dest_src/"
    cp "$REPO_ROOT/lns-backend/include/ggml-lns.h" "$dest_inc/"
}

# shellcheck source=integrate-lns.sh
source "$REPO_ROOT/scripts/integrate-lns.sh"

prepare_xlnscpp() {
    if [ ! -d "$XLNSCPP_DIR/.git" ]; then
        git -C "$REPO_ROOT" submodule update --init
    fi
    cd "$XLNSCPP_DIR"
    git fetch origin
    git reset --hard origin/main
    XLNSCPP_SHA=$(git rev-parse HEAD)
    log "    xlnscpp at $XLNSCPP_SHA"
}

fallback_xlnscpp() {
    log "    xlnscpp latest failed; using pinned commit"
    cd "$XLNSCPP_DIR"
    git reset --hard "$XLNSCPP_BASE_COMMIT"
    XLNSCPP_SHA=$(git rev-parse HEAD)
    log "    xlnscpp at $XLNSCPP_SHA"
}

run_validation() {

GGML_DIR="$BUILD_DIR/ggml"
log ""
log "==> [1/3] Unit test on latest ggml"

if [ ! -d "$GGML_DIR/.git" ]; then
    git clone https://github.com/ggml-org/ggml.git "$GGML_DIR"
fi

cd "$GGML_DIR"
git fetch origin
git reset --hard origin/master
GGML_SHA=$(git rev-parse HEAD)
log "    ggml at $GGML_SHA"

integrate_lns_regex "$GGML_DIR"
copy_backend "$GGML_DIR/src/ggml-lns" "$GGML_DIR/include"

cp "$REPO_ROOT/lns-backend/test-lns-backend.cpp" "$GGML_DIR/examples/simple/"
if ! grep -q 'test-lns-backend' "$GGML_DIR/examples/simple/CMakeLists.txt"; then
    cat >> "$GGML_DIR/examples/simple/CMakeLists.txt" << 'EOF'

if (GGML_LNS)
    add_executable(test-lns-backend test-lns-backend.cpp)
    target_link_libraries(test-lns-backend PRIVATE ggml)
endif()
EOF
fi

mkdir -p "$GGML_DIR/build-lns"
cd "$GGML_DIR/build-lns"
cmake .. \
    -DGGML_LNS=ON \
    -DGGML_METAL=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DXLNSCPP_DIR="$XLNSCPP_DIR"
cmake --build . --target test-lns-backend -j"$JOBS"

log "    Running test-lns-backend"
./bin/test-lns-backend
log "    PASS: unit test"

# ── llama.cpp inference ────────────────────────────────────────────────────────

LLAMA_DIR="$BUILD_DIR/llama.cpp"
log ""
log "==> [2/3] Inference on latest llama.cpp (SmolLM2)"

if [ ! -d "$LLAMA_DIR/.git" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi

cd "$LLAMA_DIR"
git fetch origin
git reset --hard origin/master
LLAMA_SHA=$(git rev-parse HEAD)
log "    llama.cpp at $LLAMA_SHA"

integrate_lns_regex "$LLAMA_DIR/ggml"
copy_backend "$LLAMA_DIR/ggml/src/ggml-lns" "$LLAMA_DIR/ggml/include"

mkdir -p "$LLAMA_DIR/build-lns"
cd "$LLAMA_DIR/build-lns"
cmake .. \
    -DGGML_LNS=ON \
    -DGGML_METAL=OFF \
    -DGGML_CPU_REPACK=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DXLNSCPP_DIR="$XLNSCPP_DIR"
cmake --build . --target llama-completion -j"$JOBS"

log "    Running SmolLM2 inference"
"$LLAMA_DIR/build-lns/bin/llama-completion" \
    -m "$MODEL_SMOLLM" \
    -p "The capital of France is" \
    -n 20 \
    -fa 0 \
    -no-cnv 2>&1 | tee -a "$LOG_FILE"
log "    PASS: SmolLM2 inference"

log ""
log "==> [3/3] Inference on latest llama.cpp (Llama 3.2 1B)"
log "    Running Llama 3.2 1B inference"
"$LLAMA_DIR/build-lns/bin/llama-completion" \
    -m "$MODEL_LLAMA32" \
    -p "The capital of France is" \
    -n 20 \
    -fa 0 \
    -no-cnv 2>&1 | tee -a "$LOG_FILE"
log "    PASS: Llama 3.2 1B inference"

log ""
log "==> All validation steps passed"
log "    xlnscpp:   $XLNSCPP_SHA"
log "    ggml:      $GGML_SHA"
log "    llama.cpp: $LLAMA_SHA"
log "    Log: $LOG_FILE"

# Write SHAs for patch regeneration
echo "$XLNSCPP_SHA" > "$REPO_ROOT/patches/xlnscpp-base-commit.txt.new"
echo "$GGML_SHA" > "$REPO_ROOT/patches/ggml-base-commit.txt.new"
echo "$LLAMA_SHA" > "$REPO_ROOT/patches/llama-base-commit.txt.new"
}

log "==> Preparing xlnscpp"
prepare_xlnscpp

if ! run_validation; then
    fallback_xlnscpp
    run_validation
fi
