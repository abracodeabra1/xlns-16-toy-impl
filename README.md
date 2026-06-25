# xlns-16-toy-impl

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
| [ggml](https://github.com/ggml-org/ggml) | Tensor runtime for the backend and unit test |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | End-to-end LLM inference validation |

xlnscpp is included as a git submodule for the initial checkout.  `setup.sh`
initialises it if needed, then fetches latest `origin/main`.  If the build
fails against that HEAD, it falls back to the verified-good commit in
`patches/xlnscpp-base-commit.txt`.

For manual builds without `setup.sh`, populate the submodule first:

```bash
git submodule update --init
```

This ensures `#include "xlns16.cpp"` resolves correctly in the backend.

---

## LNS Backend

The backend implements a proper `ggml_backend` (modelled on the BLAS backend)
that performs all supported ops in **xlns16** arithmetic.  It registers itself
as the `"LNS"` backend with `GGML_BACKEND_DEVICE_TYPE_ACCEL`.

Compilation uses `xlns16_table` (O(1) float↔xlns16 via a 65536-entry lookup
table) and `xlns16_alt` (reduced-branching addition path).  `xlns16_ideal` is
not defined — this matches a realistic hardware target rather than ideal
arithmetic.

### Core ggml changes (applied automatically by `setup.sh`)

| File | Change |
|---|---|
| `ggml/include/ggml.h` | Insert `GGML_TYPE_LNS16` at the current `GGML_TYPE_COUNT` value, then bump `GGML_TYPE_COUNT` by 1 |
| `ggml/src/ggml.c` | Register LNS16 type traits (`blck_size=1`, `type_size=2`, not quantized) |
| `ggml/CMakeLists.txt` | Add `GGML_LNS` CMake option (default OFF) |
| `ggml/src/CMakeLists.txt` | Add `ggml_add_backend(LNS)` |
| `ggml/src/ggml-backend-reg.cpp` | Add `#ifdef GGML_USE_LNS` include and registration |

`setup.sh` fetches latest upstream for xlnscpp (`origin/main`), ggml, and
llama.cpp (`origin/master`).  For ggml and llama.cpp it applies LNS hooks via
regex (`scripts/integrate-lns.sh`), falling back to `patches/*.patch` if needed.
Verified-good commits for all three are recorded in `patches/*-base-commit.txt`.

### Build

```bash
cd ggml/build
cmake .. -DGGML_LNS=ON
cmake --build . -j4

# Validate
./bin/test-lns-backend
```

### Validation results (MUL_MAT, xlns16_table + xlns16_alt)

Expected: max scaled relative error **~0.59%** on the reference 4×2 × 3×2
matmul.  The unit test asserts **< 0.6%** (`0.006f` tolerance in
`test-lns-backend.cpp`).

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
| `MUL_MAT` | xlns16 dot product; weights converted dynamically per row |
| `ADD` | Element-wise `xlns16_add`; broadcasting supported |
| `MUL` | Element-wise `xlns16_mul`; broadcasting supported |
| `SCALE` | `xlns16_mul` with scalar from `op_params` |
| `SOFT_MAX` | Numerically stable: max-subtract → `xlns16_exp` → sum → `xlns16_div` |
| `RMS_NORM` | Sum-of-squares in xlns16; sqrt via `xlns16_float` wrapper |
| `DIAG_MASK_INF` | Upper-triangle set to `LNS16_NEG_INF` sentinel |
| `SILU` | `xlns16_silu` per element |
| `GELU` | `xlns16_gelu` per element |
| `RELU` | `xlns16_relu` per element |
| `GET_ROWS` | Embedding lookup; dequant → F32 → xlns16 |
| `CPY` / `CONT` / `DUP` | `memcpy` or stride-aware copy; F32↔F16↔LNS16 |
| `ROPE` | cosf/sinf in F32 → xlns16 rotation; NORMAL + NEOX modes |
