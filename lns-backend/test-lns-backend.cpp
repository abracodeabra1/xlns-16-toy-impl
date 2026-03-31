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

    // Build graph
    size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(params0);
    struct ggml_cgraph  * gf = ggml_new_graph(ctx);

    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_A, rows_A);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B);

    struct ggml_tensor * result = ggml_mul_mat(ctx, a, b);
    ggml_build_forward_expand(gf, result);
    ggml_free(ctx);

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

    // Validate against expected values
    float max_rel_err = 0.0f;
    bool pass = true;
    for (int i = 0; i < 12; i++) {
        float rel_err = fabsf(out_data[i] - expected[i]) / fabsf(expected[i]);
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        if (rel_err > 0.05f) {  // 5% tolerance for xlns16
            printf("FAIL: element %d: got %.2f, expected %.2f (rel err %.2f%%)\n",
                   i, out_data[i], expected[i], rel_err * 100.0f);
            pass = false;
        }
    }

    printf("Max relative error: %.4f%%\n", max_rel_err * 100.0f);
    printf("Result: %s\n", pass ? "PASS" : "FAIL");

    ggml_backend_sched_free(sched);
    ggml_backend_free(lns_backend);
    ggml_backend_free(cpu_backend);

    return pass ? 0 : 1;
}
