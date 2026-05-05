# Reproducing the Results

## Prerequisites

- `git`, `cmake`, `g++`, `make`
- ~2 GB free disk space (llama.cpp clone + build)

## Steps

**1. Clone and initialise**
```bash
git clone https://github.com/abracodeabra1/xlns-gsoc-application
cd xlns-gsoc-application
git submodule update --init
```

**2. Download the model**

Download `SmolLM2-135M-Instruct-Q4_K_M.gguf`:

```bash
# Using huggingface-cli, in the newer versions, use the hf command instead of huggingface-cli (pip install huggingface_hub if needed)
huggingface-cli download bartowski/SmolLM2-135M-Instruct-GGUF \
    SmolLM2-135M-Instruct-Q4_K_M.gguf --local-dir .

# Or with wget
wget https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf
```

**3. Run**
```bash
# End-to-end LNS inference on SmolLM2-135M-Instruct
MODEL=/path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf ./setup.sh

# Also build and run the unit test (standalone ggml, no model needed)
./setup.sh --unit-test --skip-inference
```

`setup.sh` clones llama.cpp at a pinned commit, applies the patches in
`patches/`, copies the backend source from `lns-backend/`, and builds.
Expected output: all 30 transformer layers run without error; generated text
is incoherent relative to an FP32 baseline (expected — see proposal Section 4.7).

## What is in this repo

| Path | Contents |
|------|----------|
| `lns-backend/` | LNS backend source (`ggml-lns.cpp`, `lns-ops.cpp`, `ggml-lns.h`) and unit test |
| `challenges/` | Source code for challenges 3–5 |
| `xlnscpp/` | xlnscpp submodule (16-bit and 32-bit LNS library) |
| `patches/` | Minimal patches to llama.cpp and standalone ggml core files |
| `setup.sh` | Automated build and run script |
