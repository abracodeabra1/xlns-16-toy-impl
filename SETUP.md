# Reproducing the Results

## Prerequisites

- `git`, `cmake`, `g++`, `make`, `perl`
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
```

**3. Run**
```bash
# End-to-end LNS inference on SmolLM2 (default model path is the repo-root .gguf)
./setup.sh

# Unit test + inference on both SmolLM2 and Llama 3.2 1B
./setup.sh --unit-test

# Unit test only (no model needed)
./setup.sh --unit-test --skip-inference

# Explicit model paths
MODEL_SMOLLM=/path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf \
MODEL_LLAMA32=/path/to/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
./setup.sh
```

`MODEL=/path/to/model.gguf` remains supported as an alias for `MODEL_SMOLLM`.

## Upstream integration

`setup.sh` always fetches latest `origin/master` from [ggml](https://github.com/ggml-org/ggml) and [llama.cpp](https://github.com/ggml-org/llama.cpp), then applies LNS hooks via regex edits in [`scripts/integrate-lns.sh`](scripts/integrate-lns.sh). If regex integration or the subsequent build fails, it falls back to the pinned patches in `patches/` (verified-good commits in `patches/*-base-commit.txt`).

To re-validate against latest upstream before refreshing the fallback pins:
```bash
./scripts/validate-latest.sh
```

## What is in this repo

| Path | Contents |
|------|----------|
| `lns-backend/` | LNS backend source (`ggml-lns.cpp`, `lns-ops.cpp`, `ggml-lns.h`) and unit test |
| `challenges/` | Source code for challenges 3–5 |
| `xlnscpp/` | xlnscpp submodule (16-bit and 32-bit LNS library) |
| `patches/` | Fallback patches + pinned base commits for ggml and llama.cpp |
| `scripts/` | `integrate-lns.sh` (regex integration), `validate-latest.sh` (compatibility gate) |
| `setup.sh` | Automated build and run script |
