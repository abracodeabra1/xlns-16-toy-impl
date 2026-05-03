# xlns-gsoc-application

GSoC 2026 application code for **"Support for Logarithmic Number Systems in Large Language Models"**
(xlnsresearch / ggml + llama.cpp).

---

## Repository structure

```
challenges/
  challenge1/   Run xlns16test and xlns32test from xlnscpp (no new code written)
  challenge2/   ggml 32-bit FP matrix multiply — simple-ctx and simple-backend examples
  challenge3/   Standalone C++ FP matrix multiply (nested-loop reference)
  challenge4/   Challenge 3 repeated with internal xlns32 arithmetic
  challenge5/   Challenge 3 repeated with internal xlns16 arithmetic

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

## Challenges

### Challenge 1 — Run xlns16test and xlns32test

These programs are part of the xlnscpp library itself.  Clone xlnscpp and build:

```bash
g++ -O2 -o xlns32test xlns32test.cpp -lm && ./xlns32test
g++ -O2 -o xlns16test xlns16test.cpp -lm && ./xlns16test
```

Key observations from the output are discussed in Section 3.1 of the proposal.  The short version: `xlns32` achieves sub-0.1% relative error on all numeric tests; `xlns16` is accurate for short accumulations (dot products) but shows quantisation absorption in very long sums — a property that is acceptable and expected for LLM matrix multiply.

### Challenge 2 — ggml FP32 matrix multiplication

`simple-ctx.cpp` (no scheduler) and `simple-backend.cpp` (full backend scheduler) are unmodified ggml examples, copied here for reference.  Build them inside the ggml repository:

```bash
cd ggml && mkdir build && cd build
cmake .. && cmake --build . --target simple-ctx simple-backend -j4
./bin/simple-ctx
./bin/simple-backend
```

### Challenge 3 — Standalone FP matrix multiply

```bash
g++ -O2 -o standalone_matmult challenges/challenge3/standalone_matmult.cpp -lm
./standalone_matmult
```

Expected output (A × B^T for the 4×2 and 3×2 reference matrices):
```
  60.000  90.000  42.000
  55.000  54.000  29.000
  50.000  54.000  28.000
 110.000 126.000  64.000
```

### Challenge 4 — xlns32 internal matrix multiply

```bash
g++ -O2 -o lns_matmult_xlns32 challenges/challenge4/lns_matmult_xlns32.cpp \
    -Ixlnscpp -lm
./lns_matmult_xlns32
```

The function signature is identical to Challenge 3 (`float*` in, `float*` out).  All internal arithmetic uses `xlns32_float`; conversion is handled automatically by the xlnscpp assignment operator.

For the table-approximation variant (removes `xlns32_ideal`):
```bash
g++ -O2 -o lns_matmult_xlns32_approx challenges/challenge4/lns_matmult_xlns32_approx.cpp \
    -Ixlnscpp -lm
```

### Challenge 5 — xlns16 internal matrix multiply

```bash
g++ -O2 -o lns_matmult_xlns16 challenges/challenge5/lns_matmult_xlns16.cpp \
    -Ixlnscpp -lm
./lns_matmult_xlns16
```

Analogous to Challenge 4 but with `xlns16_float`, demonstrating the precision trade-off of 16-bit LNS.

---

## LNS Backend

The backend implements a proper `ggml_backend` (modelled on the BLAS backend) that performs all supported ops in xlns16 arithmetic.  It registers itself as the `"LNS"` backend with `GGML_BACKEND_DEVICE_TYPE_ACCEL`.

### Core ggml changes required (not included here — apply to your ggml/llama.cpp clone)

| File | Change |
|---|---|
| `ggml/include/ggml.h` | Add `GGML_TYPE_LNS16 = 41`, bump `GGML_TYPE_COUNT` to 42 |
| `ggml/src/ggml.c` | Register LNS16 type traits (blck_size=1, type_size=2, not quantized) |
| `ggml/CMakeLists.txt` | Add `GGML_LNS` CMake option (default OFF) |
| `ggml/src/CMakeLists.txt` | Add `ggml_add_backend(LNS)` |
| `ggml/src/ggml-backend-reg.cpp` | Add `#ifdef GGML_USE_LNS` include and registration |

### Build

```bash
cd ggml/build
cmake .. -DGGML_LNS=ON
cmake --build . -j4

# Validate
./bin/test-lns-backend
```

### Validation results (MUL_MAT, xlns16_table + xlns16_alt mode)

| Element | Expected (F32) | LNS Result | Rel. error |
|---------|---------------|------------|-----------|
| [0,0] | 60.00 | 59.97 | 0.05% |
| [1,0] | 55.00 | 55.00 | 0.00% |
| [2,0] | 50.00 | 49.89 | 0.22% |
| [3,0] | 110.00 | 109.99 | 0.01% |
| [0,1] | 90.00 | 89.53 | 0.52% |
| [1,1] | 54.00 | 53.82 | 0.33% |
| [2,1] | 54.00 | 53.82 | 0.33% |
| [3,1] | 126.00 | 125.26 | 0.59% |
| [0,2] | 42.00 | 41.95 | 0.12% |
| [1,2] | 29.00 | 28.87 | 0.45% |
| [2,2] | 28.00 | 27.95 | 0.18% |
| [3,2] | 64.00 | 64.00 | 0.00% |

**Max relative error: 0.59%**

### End-to-end inference (SmolLM2-135M-Instruct, Q4_K_M)

```bash
cd llama.cpp
mkdir build-lns && cd build-lns
cmake .. -DGGML_LNS=ON -DGGML_METAL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --target llama-completion -j4

./bin/llama-completion -m ../models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
    -p "The capital of France is" -n 20 -fa 0
```

The backend runs all 13 ops across 30 transformer layers without crashing.
Output quality is degraded relative to the FP baseline, which is expected: xlns16 has 7 fractional bits versus 23 for F32, and cumulative rounding errors across 30 layers cause incoherence.
This is documented in the proposal as the primary accuracy challenge and motivates the stretch goals (xlns32 accumulator, per-layer error analysis).

### Supported operations

| Op | Implementation |
|----|---------------|
| `MUL_MAT` | xlns16 dot product; weights converted dynamically per row |
| `ADD` | Element-wise `xlns16_add`; broadcasting supported |
| `MUL` | Element-wise `xlns16_mul`; broadcasting supported |
| `SCALE` | `xlns16_mul` with scalar from `op_params` |
| `SOFT_MAX` | Numerically stable: max-subtract → `xlns16_exp` → sum → `xlns16_div` |
| `RMS_NORM` | Sum-of-squares in xlns16; sqrt via `xlns16_float` wrapper |
| `DIAG_MASK_INF` | Upper-triangle set to `xlns16(-INFINITY)` |
| `SILU` | `xlns16_silu` per element |
| `GELU` | `xlns16_gelu` per element |
| `RELU` | `xlns16_relu` per element |
| `GET_ROWS` | Embedding lookup; dequant → F32 → xlns16 |
| `CPY` / `CONT` / `DUP` | `memcpy` or stride-aware copy; F32↔F16↔LNS16 |
| `ROPE` | cosf/sinf in F32 → xlns16 rotation; NORMAL + NEOX modes |

---

## Reproducing the end-to-end inference result

See [SETUP.md](SETUP.md) for full instructions.  The short version:

```bash
git clone https://github.com/abracodeabra1/xlns-gsoc-application
cd xlns-gsoc-application
git submodule update --init
MODEL=/path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf ./setup.sh
```

`setup.sh` clones llama.cpp at a pinned upstream commit, applies the patches
in `patches/`, copies the backend source, and builds everything automatically.
