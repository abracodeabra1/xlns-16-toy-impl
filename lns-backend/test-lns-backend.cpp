#include "ggml.h"
#include "ggml-backend.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// Test the LNS backend with the same matrix multiply as simple-backend.cpp

const int rows_A = 4, cols_A = 2;
float matrix_A[rows_A * cols_A] = {
    2, 8,
    5, 1,
    4, 2,
    8, 6
};

const int rows_B = 3, cols_B = 2;
float matrix_B[rows_B * cols_B] = {
    10, 5,
    9, 9,
    5, 4
};

// Expected result (A * B^T):
// [ 60  90  42
//   55  54  29
//   50  54  28
//  110 126  64 ]
// Transposed (as ggml outputs): 4 cols x 3 rows
float expected[12] = {
    60, 55, 50, 110,
    90, 54, 54, 126,
    42, 29, 28,  64
};

static float silu_ref(float x) {
    return x / (1.0f + expf(-x));
}

static bool check_close(const char * name, const std::vector<float> & out_data, const float * expected_data, float tol) {
    float max_scaled_err = 0.0f;
    bool pass = true;

    for (size_t i = 0; i < out_data.size(); i++) {
        const float abs_err = fabsf(out_data[i] - expected_data[i]);
        const float scaled_err = abs_err / fmaxf(fabsf(expected_data[i]), 1.0f);
        if (scaled_err > max_scaled_err) {
            max_scaled_err = scaled_err;
        }
        if (scaled_err > tol) {
            printf("FAIL %s: element %zu: got %.6f, expected %.6f (scaled err %.4f%%)\n",
                   name, i, out_data[i], expected_data[i], scaled_err * 100.0f);
            pass = false;
        }
    }

    printf("%s max scaled error: %.4f%%\n", name, max_scaled_err * 100.0f);
    printf("%s result: %s\n\n", name, pass ? "PASS" : "FAIL");
    return pass;
}

static ggml_context * make_context(std::vector<uint8_t> & buf) {
    size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    buf.resize(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true,
    };

    return ggml_init(params0);
}

static bool run_mul_mat_test(ggml_backend_sched_t sched) {
    std::vector<uint8_t> buf;
    struct ggml_context * ctx = make_context(buf);
    struct ggml_cgraph  * gf = ggml_new_graph(ctx);

    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_A, rows_A);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B);

    struct ggml_tensor * result = ggml_mul_mat(ctx, a, b);
    ggml_build_forward_expand(gf, result);

    // Compute
    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);

    ggml_backend_tensor_set(a, matrix_A, 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, matrix_B, 0, ggml_nbytes(b));

    ggml_backend_sched_graph_compute(sched, gf);

    // Read results
    struct ggml_tensor * out = ggml_graph_node(gf, -1);
    std::vector<float> out_data(ggml_nelements(out));
    ggml_backend_tensor_get(out, out_data.data(), 0, ggml_nbytes(out));

    // Print results
    printf("LNS mul mat (%d x %d):\n[", (int)out->ne[0], (int)out->ne[1]);
    for (int j = 0; j < out->ne[1]; j++) {
        if (j > 0) printf("\n");
        for (int i = 0; i < out->ne[0]; i++) {
            printf(" %.2f", out_data[j * out->ne[0] + i]);
        }
    }
    printf(" ]\n\n");

    bool pass = check_close("LNS mul mat", out_data, expected, 0.005f);
    ggml_free(ctx);
    return pass;
}

static bool run_swiglu_test(ggml_backend_sched_t sched, const char * name, bool split, bool swapped) {
    constexpr int rows = 2;
    constexpr int cols = 4;

    const float gate[rows * cols] = {
        -3.0f, -1.0f,  0.0f,  2.0f,
         1.5f, -0.5f,  4.0f, -2.0f,
    };
    const float up[rows * cols] = {
         0.5f, -2.0f,  3.0f,  1.0f,
        -1.0f,  4.0f,  0.25f, 2.0f,
    };
    float expected_swiglu[rows * cols];
    float fused[rows * cols * 2];

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            const int src_i = r*cols + c;
            const int fused_row = r*cols*2;
            expected_swiglu[src_i] = silu_ref(gate[src_i]) * up[src_i];
            if (swapped) {
                fused[fused_row + c]        = up[src_i];
                fused[fused_row + cols + c] = gate[src_i];
            } else {
                fused[fused_row + c]        = gate[src_i];
                fused[fused_row + cols + c] = up[src_i];
            }
        }
    }

    std::vector<uint8_t> buf;
    struct ggml_context * ctx = make_context(buf);
    struct ggml_cgraph  * gf = ggml_new_graph(ctx);

    struct ggml_tensor * result = nullptr;
    struct ggml_tensor * gate_tensor = nullptr;
    struct ggml_tensor * up_tensor = nullptr;
    struct ggml_tensor * fused_tensor = nullptr;

    if (split) {
        gate_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        up_tensor   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        result = ggml_swiglu_split(ctx, gate_tensor, up_tensor);
    } else {
        fused_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols*2, rows);
        result = swapped ? ggml_swiglu_swapped(ctx, fused_tensor) : ggml_swiglu(ctx, fused_tensor);
    }

    ggml_build_forward_expand(gf, result);

    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, gf);

    if (split) {
        ggml_backend_tensor_set(gate_tensor, gate, 0, ggml_nbytes(gate_tensor));
        ggml_backend_tensor_set(up_tensor, up, 0, ggml_nbytes(up_tensor));
    } else {
        ggml_backend_tensor_set(fused_tensor, fused, 0, ggml_nbytes(fused_tensor));
    }

    ggml_backend_sched_graph_compute(sched, gf);

    struct ggml_tensor * out = ggml_graph_node(gf, -1);
    std::vector<float> out_data(ggml_nelements(out));
    ggml_backend_tensor_get(out, out_data.data(), 0, ggml_nbytes(out));

    printf("%s (%d x %d):\n[", name, (int)out->ne[0], (int)out->ne[1]);
    for (int j = 0; j < out->ne[1]; j++) {
        if (j > 0) printf("\n");
        for (int i = 0; i < out->ne[0]; i++) {
            printf(" %.6f", out_data[j * out->ne[0] + i]);
        }
    }
    printf(" ]\n\n");

    bool pass = check_close(name, out_data, expected_swiglu, 0.01f);
    ggml_free(ctx);
    return pass;
}

int main(void) {
    ggml_time_init();

    ggml_backend_load_all();

    // Try to init the LNS backend by name
    ggml_backend_t lns_backend = ggml_backend_init_by_name("LNS", nullptr);
    if (!lns_backend) {
        fprintf(stderr, "ERROR: LNS backend not found! Make sure ggml was built with -DGGML_LNS=ON\n");
        return 1;
    }

    printf("Using backend: %s\n", ggml_backend_name(lns_backend));

    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    ggml_backend_t backends[2] = { lns_backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);

    bool pass = true;
    pass = run_mul_mat_test(sched) && pass;
    pass = run_swiglu_test(sched, "LNS swiglu split", true, false) && pass;
    pass = run_swiglu_test(sched, "LNS swiglu fused", false, false) && pass;
    pass = run_swiglu_test(sched, "LNS swiglu fused swapped", false, true) && pass;

    ggml_backend_sched_free(sched);
    ggml_backend_free(lns_backend);
    ggml_backend_free(cpu_backend);

    return pass ? 0 : 1;
}
