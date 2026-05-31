#include "lns-ops.h"
#include "ggml-impl.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

// xlns32 defines — must be set before including xlns32.cpp
// (also set via CMake target_compile_definitions; guarded here for clarity).
// Note: xlns32 has no `xlns32_table` analogue of xlns16's 65536-entry table —
// F32<->xlns32 conversion uses real log2/exp2 calls. xlns32_alt enables the
// reduced-branching sb/db addition path. xlns32_ideal is NOT defined (matches
// a realistic hardware target).
#ifndef xlns32_alt
#define xlns32_alt
#endif

#include "xlns32.cpp"

// ============================================================
// C-linkage conversion helpers (registered as type-trait hooks)
// ============================================================

extern "C" {

void lns32_to_f32_row(const void * GGML_RESTRICT src, float * GGML_RESTRICT dst, int64_t n) {
    const xlns32 * s = (const xlns32 *)src;
    for (int64_t i = 0; i < n; i++) {
        dst[i] = xlns322fp(s[i]);
    }
}

void f32_to_lns32_row(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t n) {
    xlns32_batch_from_float(src, (xlns32 *)dst, (size_t)n);
}

} // extern "C"

// ============================================================
// Helper: convert a row from any ggml type to xlns32
// Handles F32, LNS32, and any quantized type with to_float.
// ============================================================

static void row_to_xlns32(const void * src, enum ggml_type type, xlns32 * dst, float * f32_scratch, int64_t n) {
    if (type == GGML_TYPE_LNS32) {
        // Already xlns32 — direct copy, no conversion needed
        memcpy(dst, src, (size_t)n * sizeof(xlns32));
    } else if (type == GGML_TYPE_F32) {
        xlns32_batch_from_float((const float *)src, dst, (size_t)n);
    } else {
        const auto * traits = ggml_get_type_traits(type);
        GGML_ASSERT(traits->to_float != NULL);
        traits->to_float(src, f32_scratch, n);
        xlns32_batch_from_float(f32_scratch, dst, (size_t)n);
    }
}

// ============================================================
// Element-wise typed read/write helpers
// These use array indexing within a row — strides across rows
// are handled by the caller via the nb[] byte offsets.
// ============================================================

static inline xlns32 read_elem_xlns32(const void * row, int64_t i, enum ggml_type type) {
    GGML_ASSERT(type == GGML_TYPE_LNS32 || type == GGML_TYPE_F32);
    if (type == GGML_TYPE_LNS32) {
        return ((const xlns32 *)row)[i];
    }
    return fp2xlns32(((const float *)row)[i]);
}

static inline void write_elem_xlns32(void * row, int64_t i, enum ggml_type type, xlns32 val) {
    GGML_ASSERT(type == GGML_TYPE_LNS32 || type == GGML_TYPE_F32);
    if (type == GGML_TYPE_LNS32) {
        ((xlns32 *)row)[i] = val;
    } else {
        ((float *)row)[i] = xlns322fp(val);
    }
}

// ============================================================
// MUL_MAT: dst = src0^T * src1
// src0: weights (any type with to_float, or F32/LNS32)
// src1: activations (F32 or LNS32)
// dst:  F32 or LNS32 depending on dst->type
// ============================================================

void lns_mul_mat(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type0 = src0->type;
    const int64_t K = ne00;
    GGML_ASSERT(ne10 == K);

    std::vector<float>  f32_scratch(K);
    std::vector<xlns32> a_row_lns(K);
    std::vector<xlns32> b_row_lns(K);

    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            for (int64_t i1 = 0; i1 < ne11; i1++) {
                // src1 (activations): F32 or LNS32
                const void * b_row_raw = (const char *)src1->data
                    + i1*nb11 + i12*nb12 + i13*nb13;
                row_to_xlns32(b_row_raw, src1->type, b_row_lns.data(), f32_scratch.data(), K);

                for (int64_t i0 = 0; i0 < ne01; i0++) {
                    // src0 (weights): any supported type — dynamic conversion, no caching
                    const void * a_row_raw = (const char *)src0->data
                        + i0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
                    row_to_xlns32(a_row_raw, type0, a_row_lns.data(), f32_scratch.data(), K);

                    xlns32 dot = xlns32_vec_dot(a_row_lns.data(), b_row_lns.data(), K);

                    // Write result: keep as xlns32 if dst is LNS32, else convert to F32
                    void * dst_ptr = (char *)dst->data + i0*nb0 + i1*nb1 + i12*nb2 + i13*nb3;
                    if (dst->type == GGML_TYPE_LNS32) {
                        *(xlns32 *)dst_ptr = dot;
                    } else {
                        *(float *)dst_ptr = xlns322fp(dot);
                    }
                }
            }
        }
    }
}

// ============================================================
// ADD: dst = src0 + src1 (with broadcasting)
// ============================================================

void lns_add(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    // src1 broadcasts over src0: ne1x <= ne0x
    GGML_ASSERT(ne10 == ne00 || ne10 == 1);

    for (int64_t i3 = 0; i3 < ne03; i3++) {
        for (int64_t i2 = 0; i2 < ne02; i2++) {
            for (int64_t i1 = 0; i1 < ne01; i1++) {
                const void * s0 = (const char *)src0->data + i1*nb01 + i2*nb02 + i3*nb03;
                const void * s1 = (const char *)src1->data
                    + (i1 % ne11)*nb11 + (i2 % ne12)*nb12 + (i3 % ne13)*nb13;
                void * d = (char *)dst->data + i1*nb1 + i2*nb2 + i3*nb3;

                for (int64_t i0 = 0; i0 < ne00; i0++) {
                    xlns32 a = read_elem_xlns32(s0, i0, src0->type);
                    xlns32 b = read_elem_xlns32(s1, i0 % ne10, src1->type);
                    write_elem_xlns32(d, i0, dst->type, xlns32_add(a, b));
                }
            }
        }
    }
}

// ============================================================
// MUL: dst = src0 * src1 (element-wise with broadcasting)
// ============================================================

void lns_mul(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ne10 == ne00 || ne10 == 1);

    for (int64_t i3 = 0; i3 < ne03; i3++) {
        for (int64_t i2 = 0; i2 < ne02; i2++) {
            for (int64_t i1 = 0; i1 < ne01; i1++) {
                const void * s0 = (const char *)src0->data + i1*nb01 + i2*nb02 + i3*nb03;
                const void * s1 = (const char *)src1->data
                    + (i1 % ne11)*nb11 + (i2 % ne12)*nb12 + (i3 % ne13)*nb13;
                void * d = (char *)dst->data + i1*nb1 + i2*nb2 + i3*nb3;

                for (int64_t i0 = 0; i0 < ne00; i0++) {
                    xlns32 a = read_elem_xlns32(s0, i0, src0->type);
                    xlns32 b = read_elem_xlns32(s1, i0 % ne10, src1->type);
                    write_elem_xlns32(d, i0, dst->type, xlns32_mul(a, b));
                }
            }
        }
    }
}

// ============================================================
// SCALE: dst = src0 * s + b
// ============================================================

void lns_scale(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    float s, b;
    memcpy(&s, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&b, (float *) dst->op_params + 1, sizeof(float));

    const xlns32 s_lns = fp2xlns32(s);
    const xlns32 b_lns = fp2xlns32(b);

    const int64_t n = ggml_nelements(src0);

    for (int64_t i = 0; i < n; i++) {
        xlns32 val;
        if (src0->type == GGML_TYPE_LNS32) {
            val = ((const xlns32 *)src0->data)[i];
        } else {
            val = fp2xlns32(((const float *)src0->data)[i]);
        }
        val = xlns32_mul(val, s_lns);
        if (b != 0.0f) {
            val = xlns32_add(val, b_lns);
        }
        if (dst->type == GGML_TYPE_LNS32) {
            ((xlns32 *)dst->data)[i] = val;
        } else {
            ((float *)dst->data)[i] = xlns322fp(val);
        }
    }
}

// fp2xlns32(-INFINITY) is undefined behaviour: the cast of +inf to int32 saturates
// to INT_MIN on ARM/x86, producing 0xC0000000 which xlns322fp decodes as -1.0.
// Use this sentinel (sign=1, abs=max ≈ -3.4e38) wherever -inf semantics are needed.
static const xlns32 LNS32_NEG_INF = (xlns32)0xFFFFFFFFu;

// Safe float->xlns32 that maps -inf (and any value past float range) to LNS32_NEG_INF
// so that xlns32_exp(LNS32_NEG_INF - max) underflows cleanly to xlns32_zero.
static inline xlns32 float_to_lns32_safe(float v) {
    if (v <= -3.0e38f) return LNS32_NEG_INF;
    return fp2xlns32(v);
}

// ============================================================
// SOFT_MAX: dst = softmax(src0 * scale + mask)
// ============================================================

void lns_soft_max(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1]; // mask (optional)

    float scale, max_bias;
    memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));

    const int64_t ne00 = src0->ne[0]; // sequence length
    const int64_t ne01 = src0->ne[1]; // num queries
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    // For now, ignore ALiBi (max_bias). Implement basic softmax.
    for (int64_t i3 = 0; i3 < ne03; i3++) {
        for (int64_t i2 = 0; i2 < ne02; i2++) {
            for (int64_t i1 = 0; i1 < ne01; i1++) {
                const void * s0_row = (const char *)src0->data
                    + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3];
                void * d_row = (char *)dst->data
                    + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];

                // Get mask row if present (mask is always F32)
                const float * mask = nullptr;
                if (src1) {
                    const int64_t ne12 = src1->ne[2];
                    mask = (const float *)((const char *)src1->data
                        + i1*src1->nb[1] + (i2 % ne12)*src1->nb[2] + i3*src1->nb[3]);
                }

                // Step 1: scale + mask, find max.
                // Use LNS32_NEG_INF instead of fp2xlns32(-INFINITY): the latter is UB
                // and produces -1.0, causing masked positions to leak into the softmax.
                std::vector<xlns32> row(ne00);
                xlns32 max_val = LNS32_NEG_INF;
                for (int64_t i0 = 0; i0 < ne00; i0++) {
                    xlns32 v = read_elem_xlns32(s0_row, i0, src0->type);
                    float val = xlns322fp(v) * scale;
                    if (mask) val += mask[i0];
                    row[i0] = float_to_lns32_safe(val);
                    if (xlns32_gt(row[i0], max_val)) max_val = row[i0];
                }

                // Step 2: exp(x - max) and sum, all in xlns32
                xlns32 sum = fp2xlns32(0.0f);
                for (int64_t i0 = 0; i0 < ne00; i0++) {
                    xlns32 diff = xlns32_sub(row[i0], max_val);
                    row[i0] = xlns32_exp(diff);
                    sum = xlns32_add(sum, row[i0]);
                }

                // Step 3: normalize and write
                for (int64_t i0 = 0; i0 < ne00; i0++) {
                    write_elem_xlns32(d_row, i0, dst->type, xlns32_div(row[i0], sum));
                }
            }
        }
    }
}

// ============================================================
// RMS_NORM: dst = x / sqrt(mean(x^2) + eps)
// ============================================================

void lns_rms_norm(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    for (int64_t i3 = 0; i3 < ne03; i3++) {
        for (int64_t i2 = 0; i2 < ne02; i2++) {
            for (int64_t i1 = 0; i1 < ne01; i1++) {
                const void * s = (const char *)src0->data
                    + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3];
                void * d = (char *)dst->data
                    + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];

                // sum of squares in xlns32
                xlns32 sum_sq = fp2xlns32(0.0f);
                for (int64_t i0 = 0; i0 < ne00; i0++) {
                    xlns32 x = read_elem_xlns32(s, i0, src0->type);
                    sum_sq = xlns32_add(sum_sq, xlns32_mul(x, x));
                }

                // mean = sum_sq / n
                xlns32 mean = xlns32_div(sum_sq, fp2xlns32((float)ne00));

                // rms = sqrt(mean + eps)
                xlns32 rms = xlns32_add(mean, fp2xlns32(eps));
                xlns32_float rms_f = float2xlns32_(xlns322fp(rms));
                xlns32_float sqrt_rms = sqrt(rms_f);

                // inv_rms = 1 / sqrt_rms
                xlns32 inv_rms = xlns32_div(fp2xlns32(1.0f), xlns32_internal(sqrt_rms));

                // normalize: d[i] = s[i] * inv_rms
                for (int64_t i0 = 0; i0 < ne00; i0++) {
                    xlns32 x = read_elem_xlns32(s, i0, src0->type);
                    write_elem_xlns32(d, i0, dst->type, xlns32_mul(x, inv_rms));
                }
            }
        }
    }
}

// ============================================================
// DIAG_MASK_INF: causal attention mask
// ============================================================

void lns_diag_mask_inf(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    int n_past;
    memcpy(&n_past, dst->op_params, sizeof(int32_t));

    const int64_t nc = src0->ne[0]; // columns
    const int64_t nr = src0->ne[1]; // rows
    const int64_t n  = ggml_nrows(src0);
    const int64_t nz = n / nr;

    const bool inplace = (src0->data == dst->data);
    if (!inplace) {
        // ggml_nbytes uses type, which is already updated by the pre-pass
        GGML_ASSERT(ggml_nbytes(dst) <= ggml_nbytes(src0)); // safety: xlns32 is same width as F32
        memcpy(dst->data, src0->data, ggml_nbytes(dst));
    }

    if (dst->type == GGML_TYPE_LNS32) {
        const xlns32 neg_inf = fp2xlns32(-INFINITY);
        xlns32 * data = (xlns32 *)dst->data;
        for (int64_t k = 0; k < nz; k++) {
            for (int64_t j = 0; j < nr; j++) {
                for (int64_t i = n_past + j + 1; i < nc; i++) {
                    data[k*nr*nc + j*nc + i] = neg_inf;
                }
            }
        }
    } else {
        float * data = (float *)dst->data;
        for (int64_t k = 0; k < nz; k++) {
            for (int64_t j = 0; j < nr; j++) {
                for (int64_t i = n_past + j + 1; i < nc; i++) {
                    data[k*nr*nc + j*nc + i] = -INFINITY;
                }
            }
        }
    }
}

// ============================================================
// SILU: dst = x * sigmoid(x)
// ============================================================

void lns_silu(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const int64_t n = ggml_nelements(src0);

    for (int64_t i = 0; i < n; i++) {
        xlns32 v = (src0->type == GGML_TYPE_LNS32)
            ? ((const xlns32 *)src0->data)[i]
            : fp2xlns32(((const float *)src0->data)[i]);
        xlns32 r = xlns32_silu(v);
        if (dst->type == GGML_TYPE_LNS32) {
            ((xlns32 *)dst->data)[i] = r;
        } else {
            ((float *)dst->data)[i] = xlns322fp(r);
        }
    }
}

// ============================================================
// GELU: dst = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
// ============================================================

void lns_gelu(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const int64_t n = ggml_nelements(src0);

    for (int64_t i = 0; i < n; i++) {
        xlns32 v = (src0->type == GGML_TYPE_LNS32)
            ? ((const xlns32 *)src0->data)[i]
            : fp2xlns32(((const float *)src0->data)[i]);
        xlns32 r = xlns32_gelu(v);
        if (dst->type == GGML_TYPE_LNS32) {
            ((xlns32 *)dst->data)[i] = r;
        } else {
            ((float *)dst->data)[i] = xlns322fp(r);
        }
    }
}

// ============================================================
// RELU: dst = max(0, x)
// ============================================================

void lns_relu(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const int64_t n = ggml_nelements(src0);

    for (int64_t i = 0; i < n; i++) {
        xlns32 v = (src0->type == GGML_TYPE_LNS32)
            ? ((const xlns32 *)src0->data)[i]
            : fp2xlns32(((const float *)src0->data)[i]);
        xlns32 r = xlns32_relu(v);
        if (dst->type == GGML_TYPE_LNS32) {
            ((xlns32 *)dst->data)[i] = r;
        } else {
            ((float *)dst->data)[i] = xlns322fp(r);
        }
    }
}

// ============================================================
// GET_ROWS: gather rows from src0 using indices in src1
// ============================================================

void lns_get_rows(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0]; // embedding table
    const struct ggml_tensor * src1 = dst->src[1]; // indices (I32)

    const int64_t nc = src0->ne[0]; // embedding dim
    const enum ggml_type type = src0->type;
    const auto * traits = ggml_get_type_traits(type);

    const int64_t nr = ggml_nelements(src1); // number of rows to gather

    std::vector<float> f32_scratch(nc);

    for (int64_t i = 0; i < nr; i++) {
        // Compute multi-dimensional index into src1
        const int64_t ne10 = src1->ne[0];
        const int64_t ne11 = src1->ne[1];
        const int64_t i12 = i / (ne11 * ne10);
        const int64_t i11 = (i - i12*ne11*ne10) / ne10;
        const int64_t i10 = i - i12*ne11*ne10 - i11*ne10;

        // Get row index
        const int32_t row_idx = ((const int32_t *)((const char *)src1->data
            + i10*src1->nb[0] + i11*src1->nb[1] + i12*src1->nb[2]))[0];
        GGML_ASSERT(row_idx >= 0 && row_idx < src0->ne[1]);

        // Source row (embedding table — never LNS32, always a weight type)
        const void * src_row = (const char *)src0->data
            + row_idx*src0->nb[1] + i11*src0->nb[2] + i12*src0->nb[3];

        if (dst->type == GGML_TYPE_LNS32) {
            // Output directly as xlns32: dequantize to F32, then convert
            xlns32 * dst_row = (xlns32 *)((char *)dst->data
                + i10*dst->nb[1] + i11*dst->nb[2] + i12*dst->nb[3]);
            if (type == GGML_TYPE_F32) {
                xlns32_batch_from_float((const float *)src_row, dst_row, (size_t)nc);
            } else {
                GGML_ASSERT(traits->to_float != NULL);
                traits->to_float(src_row, f32_scratch.data(), nc);
                xlns32_batch_from_float(f32_scratch.data(), dst_row, (size_t)nc);
            }
        } else {
            // Output as F32 (original behaviour, for non-LNS mode)
            float * dst_row = (float *)((char *)dst->data
                + i10*dst->nb[1] + i11*dst->nb[2] + i12*dst->nb[3]);
            if (type == GGML_TYPE_F32) {
                memcpy(dst_row, src_row, (size_t)nc * sizeof(float));
            } else {
                GGML_ASSERT(traits->to_float != NULL);
                traits->to_float(src_row, dst_row, nc);
            }
        }
    }
}

// ============================================================
// CPY / CONT / DUP: copy src0 to dst (possibly with type conversion)
// ============================================================

void lns_cpy(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    // Same-type contiguous tensors: raw memcpy
    if (src0->type == dst->type && ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
        memcpy(dst->data, src0->data, ggml_nbytes(dst));
        return;
    }

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    // F32 -> F32 (with stride differences)
    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        for (int64_t i3 = 0; i3 < ne03; i3++)
            for (int64_t i2 = 0; i2 < ne02; i2++)
                for (int64_t i1 = 0; i1 < ne01; i1++) {
                    const float * s = (const float *)((const char *)src0->data
                        + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3]);
                    float * d = (float *)((char *)dst->data
                        + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3]);
                    memcpy(d, s, (size_t)ne00 * sizeof(float));
                }
        return;
    }

    // F32 -> F16 (common for KV cache)
    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        for (int64_t i3 = 0; i3 < ne03; i3++)
            for (int64_t i2 = 0; i2 < ne02; i2++)
                for (int64_t i1 = 0; i1 < ne01; i1++) {
                    const float * s = (const float *)((const char *)src0->data
                        + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3]);
                    ggml_fp16_t * d = (ggml_fp16_t *)((char *)dst->data
                        + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3]);
                    for (int64_t i0 = 0; i0 < ne00; i0++) {
                        d[i0] = ggml_fp32_to_fp16(s[i0]);
                    }
                }
        return;
    }

    // LNS32 -> F32 (e.g., output logits conversion)
    if (src0->type == GGML_TYPE_LNS32 && dst->type == GGML_TYPE_F32) {
        for (int64_t i3 = 0; i3 < ne03; i3++)
            for (int64_t i2 = 0; i2 < ne02; i2++)
                for (int64_t i1 = 0; i1 < ne01; i1++) {
                    const xlns32 * s = (const xlns32 *)((const char *)src0->data
                        + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3]);
                    float * d = (float *)((char *)dst->data
                        + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3]);
                    lns32_to_f32_row(s, d, ne00);
                }
        return;
    }

    // F32 -> LNS32
    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_LNS32) {
        for (int64_t i3 = 0; i3 < ne03; i3++)
            for (int64_t i2 = 0; i2 < ne02; i2++)
                for (int64_t i1 = 0; i1 < ne01; i1++) {
                    const float * s = (const float *)((const char *)src0->data
                        + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3]);
                    xlns32 * d = (xlns32 *)((char *)dst->data
                        + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3]);
                    f32_to_lns32_row(s, d, ne00);
                }
        return;
    }

    // LNS32 -> LNS32 (with stride differences — same-type contiguous handled above)
    if (src0->type == GGML_TYPE_LNS32 && dst->type == GGML_TYPE_LNS32) {
        for (int64_t i3 = 0; i3 < ne03; i3++)
            for (int64_t i2 = 0; i2 < ne02; i2++)
                for (int64_t i1 = 0; i1 < ne01; i1++) {
                    const xlns32 * s = (const xlns32 *)((const char *)src0->data
                        + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3]);
                    xlns32 * d = (xlns32 *)((char *)dst->data
                        + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3]);
                    memcpy(d, s, (size_t)ne00 * sizeof(xlns32));
                }
        return;
    }

    GGML_ABORT("lns_cpy: unsupported type conversion %s -> %s\n",
               ggml_type_name(src0->type), ggml_type_name(dst->type));
}

// ============================================================
// ROPE: Rotary Position Embeddings
// ============================================================

void lns_rope(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0]; // Q or K tensor
    const struct ggml_tensor * src1 = dst->src[1]; // position indices (I32)

    const int64_t ne0 = src0->ne[0]; // head dim
    const int64_t ne1 = src0->ne[1]; // num heads
    const int64_t ne2 = src0->ne[2]; // seq len
    const int64_t ne3 = src0->ne[3]; // batch

    const int32_t n_dims     = ((int32_t *) dst->op_params)[1];
    const int32_t mode       = ((int32_t *) dst->op_params)[2];
    const int32_t n_ctx_orig = ((int32_t *) dst->op_params)[4];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));

    const bool is_neox = (mode & 2) != 0; // NEOX-style: pairs at distance n_dims/2

    (void)n_ctx_orig; (void)ext_factor; (void)attn_factor;
    (void)beta_fast; (void)beta_slow; (void)freq_scale;

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            const int32_t pos = ((const int32_t *)src1->data)[i2];
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                const void * src_row = (const char *)src0->data
                    + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3];
                void * dst_row = (char *)dst->data
                    + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];

                float t = (float)pos;
                for (int64_t j = 0; j < n_dims; j += 2) {
                    float cos_t = cosf(t);
                    float sin_t = sinf(t);

                    // Convert cos/sin to xlns32 for the rotation
                    xlns32 c     = fp2xlns32(cos_t);
                    xlns32 s_val = fp2xlns32(sin_t);

                    int64_t i0_0, i0_1;
                    if (is_neox) {
                        // NEOX: pairs at (j/2, j/2 + n_dims/2)
                        i0_0 = j / 2;
                        i0_1 = j / 2 + n_dims / 2;
                    } else {
                        // Normal: adjacent pairs (j, j+1)
                        i0_0 = j;
                        i0_1 = j + 1;
                    }

                    xlns32 x0 = read_elem_xlns32(src_row, i0_0, src0->type);
                    xlns32 x1 = read_elem_xlns32(src_row, i0_1, src0->type);

                    // Rotation: [cos -sin; sin cos] * [x0; x1]
                    xlns32 r0 = xlns32_sub(xlns32_mul(x0, c), xlns32_mul(x1, s_val));
                    xlns32 r1 = xlns32_add(xlns32_mul(x0, s_val), xlns32_mul(x1, c));

                    write_elem_xlns32(dst_row, i0_0, dst->type, r0);
                    write_elem_xlns32(dst_row, i0_1, dst->type, r1);

                    t *= theta_scale;
                }

                // Copy non-rotated dimensions (passthrough in xlns32 or F32)
                for (int64_t j = n_dims; j < ne0; j++) {
                    xlns32 v = read_elem_xlns32(src_row, j, src0->type);
                    write_elem_xlns32(dst_row, j, dst->type, v);
                }
            }
        }
    }
}
