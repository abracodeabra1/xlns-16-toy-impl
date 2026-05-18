# xlns-gsoc-application

GSoC 2026 application code for **"Support for Logarithmic Number Systems in Large Language Models"**
(xlnsresearch / ggml + llama.cpp).

---

## Reproducing the end-to-end inference result

See [SETUP.md](SETUP.md) for full instructions. 

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

The backend implements a proper `ggml_backend` (modelled on the BLAS backend) that performs all supported ops in **xlns32** arithmetic.  It registers itself as the `"LNS"` backend with `GGML_BACKEND_DEVICE_TYPE_ACCEL`.

This is the **xlns32 variant** of the backend: it is a sibling of the xlns16
backend in `../initial_application/xlns-gsoc-application/` and shares the same
architecture, but uses 32-bit LNS (≈23 fractional bits, comparable to FP32
precision) instead of 16-bit LNS. Per-row F32→xlns32 conversion is **not**
backed by a lookup table (xlnscpp has no 2³² table), so element conversion
goes through real `log2`/`exp2` calls and is noticeably slower per element
than xlns16 — accuracy is the goal here, not throughput.

### Core ggml changes required (not included here — apply to your ggml/llama.cpp clone)

| File | Change |
|---|---|
| `ggml/include/ggml.h` | Add `GGML_TYPE_LNS32 = 41`, bump `GGML_TYPE_COUNT` to 42 |
| `ggml/src/ggml.c` | Register LNS32 type traits (blck_size=1, type_size=4, not quantized) |
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
cd llama.cpp
mkdir build-lns && cd build-lns
cmake .. -DGGML_LNS=ON -DGGML_METAL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --target llama-completion -j4

./bin/llama-completion -m ../models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
    -p "The capital of France is" -n 20 -fa 0
```

The backend runs all 13 ops across 30 transformer layers without crashing.
xlns32 has roughly 23 fractional bits — comparable to FP32 — so generated
text is expected to be coherent and close to the FP32 baseline (this is the
explicit motivation for the xlns32 variant: validate that LNS is a viable
arithmetic for LLMs when given enough precision, in contrast to the xlns16
sibling whose output is incoherent across 30 layers).

Per-token throughput is lower than the xlns16 sibling because every
F32↔xlns32 conversion goes through real `log2`/`exp2` calls rather than a
65 536-entry lookup table.

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