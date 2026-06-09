#!/usr/bin/env bash
# setup.sh — reproduce the LNS backend results from scratch.
#
# Usage:
#   ./setup.sh                    # end-to-end inference (requires at least one model)
#   ./setup.sh --unit-test        # also build and run test-lns-backend (standalone ggml)
#   ./setup.sh --unit-test --skip-inference
#
# Models (at least one required for inference):
#   MODEL_SMOLLM  — SmolLM2-135M-Instruct-Q4_K_M.gguf (in repo root by default)
#   MODEL_LLAMA32   — Llama-3.2-1B-Instruct-Q4_K_M.gguf
#   MODEL           — alias for MODEL_SMOLLM (backward compatible)
#
# Upstream integration: always fetches origin/master, applies regex edits first,
# falls back to pinned patches in patches/ if regex or build fails.

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
XLNSCPP_DIR="$REPO_ROOT/xlnscpp"
BUILD_DIR="$REPO_ROOT/build"
LLAMA_BASE_COMMIT="$(cat "$REPO_ROOT/patches/llama-base-commit.txt")"
GGML_BASE_COMMIT="$(cat "$REPO_ROOT/patches/ggml-base-commit.txt")"

MODEL_SMOLLM="${MODEL_SMOLLM:-${MODEL:-$REPO_ROOT/SmolLM2-135M-Instruct-Q4_K_M.gguf}}"
MODEL_LLAMA32="${MODEL_LLAMA32:-}"

RUN_UNIT_TEST=0
RUN_INFERENCE=1

for arg in "$@"; do
    case "$arg" in
        --unit-test)         RUN_UNIT_TEST=1 ;;
        --skip-inference)    RUN_INFERENCE=0 ;;
    esac
done

resolve_model_path() {
    local var_name="$1"
    local path="$2"
    if [ -n "$path" ]; then
        if [ ! -f "$path" ]; then
            echo "ERROR: $var_name model not found: $path" >&2
            exit 1
        fi
        echo "$(cd "$(dirname "$path")" && pwd)/$(basename "$path")"
    fi
}

MODEL_SMOLLM="$(resolve_model_path MODEL_SMOLLM "$MODEL_SMOLLM")"
MODEL_LLAMA32="$(resolve_model_path MODEL_LLAMA32 "$MODEL_LLAMA32")"

if [ "$RUN_INFERENCE" -eq 1 ] && [ -z "$MODEL_SMOLLM" ] && [ -z "$MODEL_LLAMA32" ]; then
    echo "ERROR: At least one model path required for inference." >&2
    echo "  MODEL_SMOLLM=/path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf ./setup.sh" >&2
    echo "  MODEL_LLAMA32=/path/to/Llama-3.2-1B-Instruct-Q4_K_M.gguf ./setup.sh" >&2
    echo "" >&2
    echo "SmolLM2 is included in the repo root:" >&2
    echo "  MODEL_SMOLLM=$REPO_ROOT/SmolLM2-135M-Instruct-Q4_K_M.gguf ./setup.sh" >&2
    exit 1
fi

# Detect parallel job count
if command -v nproc &>/dev/null; then
    JOBS=$(nproc)
else
    JOBS=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
fi

# shellcheck source=scripts/integrate-lns.sh
source "$REPO_ROOT/scripts/integrate-lns.sh"

check_deps() {
    for cmd in git cmake g++ make perl; do
        if ! command -v "$cmd" &>/dev/null; then
            echo "ERROR: '$cmd' not found. Please install it and retry." >&2
            exit 1
        fi
    done
}

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

fetch_latest() {
    local repo_dir="$1"
    cd "$repo_dir"
    git fetch origin
    git reset --hard origin/master
    echo "    at $(git rev-parse --short HEAD)"
}

prepare_ggml_test_files() {
    local ggml_dir="$1"
    cp "$REPO_ROOT/lns-backend/test-lns-backend.cpp" "$ggml_dir/examples/simple/"
    if ! grep -q 'test-lns-backend' "$ggml_dir/examples/simple/CMakeLists.txt"; then
        cat >> "$ggml_dir/examples/simple/CMakeLists.txt" << 'EOF'

if (GGML_LNS)
    add_executable(test-lns-backend test-lns-backend.cpp)
    target_link_libraries(test-lns-backend PRIVATE ggml)
endif()
EOF
    fi
}

prepare_ggml() {
    local ggml_dir="$1"
    if [ ! -d "$ggml_dir/.git" ]; then
        echo "    Cloning ggml..."
        git clone https://github.com/ggml-org/ggml.git "$ggml_dir"
    fi
    echo "    Fetching latest ggml"
    fetch_latest "$ggml_dir"
    integrate_lns_or_fallback "$ggml_dir" \
        "$REPO_ROOT/patches/ggml-core.patch" "$GGML_BASE_COMMIT"
    copy_backend "$ggml_dir/src/ggml-lns" "$ggml_dir/include"
}

prepare_llama() {
    local llama_dir="$1"
    if [ ! -d "$llama_dir/.git" ]; then
        echo "    Cloning llama.cpp (this may take a few minutes)..."
        git clone https://github.com/ggml-org/llama.cpp.git "$llama_dir"
    fi
    echo "    Fetching latest llama.cpp"
    fetch_latest "$llama_dir"
    integrate_lns_or_fallback "$llama_dir" \
        "$REPO_ROOT/patches/llama-core.patch" "$LLAMA_BASE_COMMIT"
    copy_backend "$llama_dir/ggml/src/ggml-lns" "$llama_dir/ggml/include"
}

run_inference() {
    local label="$1"
    local model_path="$2"
    local llama_bin="$3"

    echo ""
    echo "==> Running LNS inference: $label"
    "$llama_bin" \
        -m "$model_path" \
        -p "The capital of France is" \
        -n 20 \
        -fa 0 \
        -no-cnv
}

# ── Prerequisites ─────────────────────────────────────────────────────────────

check_deps

echo "==> Initialising xlnscpp submodule"
git -C "$REPO_ROOT" submodule update --init

mkdir -p "$BUILD_DIR"

# ── Unit test (standalone ggml) ───────────────────────────────────────────────

if [ "$RUN_UNIT_TEST" -eq 1 ]; then
    echo ""
    echo "==> Setting up standalone ggml for unit test"

    GGML_DIR="$BUILD_DIR/ggml"
    prepare_ggml "$GGML_DIR"
    prepare_ggml_test_files "$GGML_DIR"

    echo "    Building"
    mkdir -p "$GGML_DIR/build-lns"
    cd "$GGML_DIR/build-lns"
    cmake .. \
        -DGGML_LNS=ON \
        -DGGML_METAL=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DXLNSCPP_DIR="$XLNSCPP_DIR"
    cmake --build . --target test-lns-backend -j"$JOBS"

    echo ""
    echo "==> Running test-lns-backend"
    ./bin/test-lns-backend
fi

# ── End-to-end inference (llama.cpp) ─────────────────────────────────────────

if [ "$RUN_INFERENCE" -eq 1 ]; then
    echo ""
    echo "==> Setting up llama.cpp for end-to-end LNS inference"

    LLAMA_DIR="$BUILD_DIR/llama.cpp"
    prepare_llama "$LLAMA_DIR"

    echo "    Building llama-completion"
    mkdir -p "$LLAMA_DIR/build-lns"
    cd "$LLAMA_DIR/build-lns"
    cmake .. \
        -DGGML_LNS=ON \
        -DGGML_METAL=OFF \
        -DGGML_CPU_REPACK=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DXLNSCPP_DIR="$XLNSCPP_DIR"
    cmake --build . --target llama-completion -j"$JOBS"

    LLAMA_BIN="$LLAMA_DIR/build-lns/bin/llama-completion"

    if [ -n "$MODEL_SMOLLM" ]; then
        run_inference "SmolLM2-135M-Instruct" "$MODEL_SMOLLM" "$LLAMA_BIN"
    fi
    if [ -n "$MODEL_LLAMA32" ]; then
        run_inference "Llama-3.2-1B-Instruct" "$MODEL_LLAMA32" "$LLAMA_BIN"
    fi
fi

echo ""
echo "==> Done."
