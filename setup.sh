#!/usr/bin/env bash
# setup.sh — reproduce the LNS backend results from scratch.
#
# Usage:
#   ./setup.sh                    # end-to-end inference only (requires model, see below)
#   ./setup.sh --unit-test        # also build and run test-lns-backend (standalone ggml)
#   ./setup.sh --unit-test --skip-inference
#
# Model required for inference:
#   SmolLM2-135M-Instruct-Q4_K_M.gguf — download with:
#     huggingface-cli download bartowski/SmolLM2-135M-Instruct-GGUF \
#         SmolLM2-135M-Instruct-Q4_K_M.gguf --local-dir .
#   Then pass the path as: MODEL=/path/to/model.gguf ./setup.sh

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
XLNSCPP_DIR="$REPO_ROOT/xlnscpp"
BUILD_DIR="$REPO_ROOT/build"
LLAMA_BASE_COMMIT="$(cat "$REPO_ROOT/patches/llama-base-commit.txt")"
GGML_BASE_COMMIT="$(cat "$REPO_ROOT/patches/ggml-base-commit.txt")"
MODEL="${MODEL:-}"

RUN_UNIT_TEST=0
RUN_INFERENCE=1

for arg in "$@"; do
    case "$arg" in
        --unit-test)         RUN_UNIT_TEST=1 ;;
        --skip-inference)    RUN_INFERENCE=0 ;;
    esac
done

if [ "$RUN_INFERENCE" -eq 1 ] && [ -z "$MODEL" ]; then
    echo "ERROR: MODEL path not set. Download the model and run:"
    echo "  MODEL=/path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf ./setup.sh"
    echo ""
    echo "Download the model with:"
    echo "  huggingface-cli download bartowski/SmolLM2-135M-Instruct-GGUF \\"
    echo "      SmolLM2-135M-Instruct-Q4_K_M.gguf --local-dir ."
    exit 1
fi

# Detect parallel job count
if command -v nproc &>/dev/null; then
    JOBS=$(nproc)
else
    JOBS=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
fi

# ── Helpers ──────────────────────────────────────────────────────────────────

check_deps() {
    for cmd in git cmake g++ make; do
        if ! command -v "$cmd" &>/dev/null; then
            echo "ERROR: '$cmd' not found. Please install it and retry." >&2
            exit 1
        fi
    done
}

copy_backend() {
    local dest_src="$1"   # e.g. repo/ggml/src/ggml-lns
    local dest_inc="$2"   # e.g. repo/ggml/include
    mkdir -p "$dest_src"
    cp "$REPO_ROOT/lns-backend/src/ggml-lns.cpp" "$dest_src/"
    cp "$REPO_ROOT/lns-backend/src/lns-ops.cpp"  "$dest_src/"
    cp "$REPO_ROOT/lns-backend/src/lns-ops.h"    "$dest_src/"
    cp "$REPO_ROOT/lns-backend/src/CMakeLists.txt" "$dest_src/"
    cp "$REPO_ROOT/lns-backend/include/ggml-lns.h" "$dest_inc/"
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

    if [ ! -d "$GGML_DIR/.git" ]; then
        echo "    Cloning ggml..."
        git clone https://github.com/ggml-org/ggml.git "$GGML_DIR"
    fi

    cd "$GGML_DIR"
    echo "    Checking out base commit $GGML_BASE_COMMIT"
    git checkout "$GGML_BASE_COMMIT"

    echo "    Applying ggml-core.patch"
    git apply "$REPO_ROOT/patches/ggml-core.patch"

    echo "    Copying backend source files"
    copy_backend "$GGML_DIR/src/ggml-lns" "$GGML_DIR/include"

    echo "    Copying test-lns-backend.cpp"
    cp "$REPO_ROOT/lns-backend/test-lns-backend.cpp" "$GGML_DIR/examples/simple/"
    cat >> "$GGML_DIR/examples/simple/CMakeLists.txt" << 'EOF'

if (GGML_LNS)
    add_executable(test-lns-backend test-lns-backend.cpp)
    target_link_libraries(test-lns-backend PRIVATE ggml)
endif()
EOF

    echo "    Building"
    mkdir -p "$GGML_DIR/build-lns"
    cd "$GGML_DIR/build-lns"
    cmake .. \
        -DGGML_LNS=ON \
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

    if [ ! -d "$LLAMA_DIR/.git" ]; then
        echo "    Cloning llama.cpp (this may take a few minutes)..."
        git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
    fi

    cd "$LLAMA_DIR"
    echo "    Checking out base commit $LLAMA_BASE_COMMIT"
    git checkout "$LLAMA_BASE_COMMIT"

    echo "    Applying llama-core.patch"
    git apply "$REPO_ROOT/patches/llama-core.patch"

    echo "    Copying backend source files"
    copy_backend "$LLAMA_DIR/ggml/src/ggml-lns" "$LLAMA_DIR/ggml/include"

    echo "    Building llama-completion"
    mkdir -p "$LLAMA_DIR/build-lns"
    cd "$LLAMA_DIR/build-lns"
    cmake .. \
        -DGGML_LNS=ON \
        -DGGML_METAL=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DXLNSCPP_DIR="$XLNSCPP_DIR"
    cmake --build . --target llama-completion -j"$JOBS"

    echo ""
    echo "==> Running LNS inference on SmolLM2-135M-Instruct"

    "$LLAMA_DIR/build-lns/bin/llama-completion" \
        -m "$MODEL" \
        -p "The capital of France is" \
        -n 20 \
        -fa 0
fi

echo ""
echo "==> Done."
