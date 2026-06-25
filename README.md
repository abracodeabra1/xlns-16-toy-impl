# xlns-gsoc-application

GSoC 2026 application code for **"Support for Logarithmic Number Systems in Large Language Models"**
(xlnsresearch / ggml + llama.cpp).

---

## Reproducing the end-to-end inference result

See [SETUP.md](SETUP.md) for full instructions. 

---

## Repository structure

```

lns-backend/
  include/      Public API header  (ggml-lns.h)
  src/          Backend implementation (ggml-lns.cpp, lns-ops.h, lns-ops.cpp, CMakeLists.txt)
  test-lns-backend.cpp   Validation test (same matrix as challenge 2)
```

---

## Dependencies

| Dependency | Purpose |
|---|---|
| [xlnscpp](https://github.com/xlnsresearch/xlnscpp) | C++ LNS library — included as a **git submodule** |
| [ggml](https://github.com/ggml-org/ggml) | Required by challenge 2 and the full backend |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Required for end-to-end LLM inference validation |

xlnscpp is included as a submodule.  After cloning this repo, run:

```bash
git submodule update --init
```

This populates `xlnscpp/` so that `#include "xlns16.cpp"` and `#include "xlns32.cpp"` resolve correctly in challenges 4/5 and the backend.

---


## LNS Backend

The backend implements a proper `ggml_backend` (modelled on the BLAS backend) that performs all supported ops in **xlns32** arithmetic.  It registers itself as the `"LNS"` backend with `GGML_BACKEND_DEVICE_TYPE_ACCEL`.

This is the **xlns32 variant** of the backend: it is a sibling of the xlns16
backend in `../initial_application/xlns-gsoc-application/` and shares the same
architecture, but uses 32-bit LNS (≈23 fractional bits, comparable to FP32
precision) instead of 16-bit LNS. Per-row F32→xlns32 conversion is **not**
backed by a lookup table (xlnscpp has no 2³² table), so element conversion
goes through real `log2`/`exp2` calls and is noticeably slower per element
than xlns16 — accuracy is the goal here, not throughput.

### Core ggml changes (applied automatically by `setup.sh`)

| File | Change |
|---|---|
| `ggml/include/ggml.h` | Insert `GGML_TYPE_LNS32` at the current `GGML_TYPE_COUNT` value, then bump `GGML_TYPE_COUNT` by 1 |
| `ggml/src/ggml.c` | Register LNS32 type traits (blck_size=1, type_size=4, not quantized) |
| `ggml/CMakeLists.txt` | Add `GGML_LNS` CMake option (default OFF) |
| `ggml/src/CMakeLists.txt` | Add `ggml_add_backend(LNS)` |
| `ggml/src/ggml-backend-reg.cpp` | Add `#ifdef GGML_USE_LNS` include and registration |

`setup.sh` fetches latest upstream and applies these via regex (`scripts/integrate-lns.sh`), falling back to `patches/*.patch` if needed. Fallback pins are recorded in `patches/*-base-commit.txt`.

### Build

```bash
cd ggml/build
cmake .. -DGGML_LNS=ON
cmake --build . -j4

# Validate
./bin/test-lns-backend
```

### Validation results (MUL_MAT, xlns32_alt mode)

Expected: max relative error well under 0.1% on the reference 4×2 × 3×2 matmul
(Challenge 4's stand-alone xlns32 matmul achieves the same). The unit test
asserts < 0.5% to leave headroom; concrete per-element numbers will be filled
in after the first local build.

> Compare to the xlns16 backend's max relative error of 0.59% on the same
> matmul — xlns32 adds ~16 more fractional bits of precision, eliminating
> nearly all of that error.

### End-to-end inference (SmolLM2-135M-Instruct, Q4_K_M)

```bash
# SmolLM2 .gguf is in the repo root; setup.sh uses it by default
./setup.sh

# Or with Llama 3.2 1B as well
MODEL_LLAMA32=./Llama-3.2-1B-Instruct-Q4_K_M.gguf ./setup.sh
```


### Supported operations

| Op | Implementation |
|----|---------------|
| `MUL_MAT` | xlns32 dot product; weights converted dynamically per row |
| `ADD` | Element-wise `xlns32_add`; broadcasting supported |
| `MUL` | Element-wise `xlns32_mul`; broadcasting supported |
| `SCALE` | `xlns32_mul` with scalar from `op_params` |
| `SOFT_MAX` | Numerically stable: max-subtract → `xlns32_exp` → sum → `xlns32_div` |
| `RMS_NORM` | Sum-of-squares in xlns32; sqrt via `xlns32_float` wrapper |
| `DIAG_MASK_INF` | Upper-triangle set to `xlns32(-INFINITY)` |
| `SILU` | `xlns32_silu` per element |
| `GELU` | `xlns32_gelu` per element |
| `RELU` | `xlns32_relu` per element |
| `GET_ROWS` | Embedding lookup; dequant → F32 → xlns32 |
| `CPY` / `CONT` / `DUP` | `memcpy` or stride-aware copy; F32↔F16↔LNS32 |
| `ROPE` | cosf/sinf in F32 → xlns32 rotation; NORMAL + NEOX modes |