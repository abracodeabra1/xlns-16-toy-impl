//#define xlns32_ideal   // accurate Gaussian log, comment out later to test approx
#include "xlns32.cpp"
#include <stdio.h>

#include <stdlib.h>
#include <iostream>

void mat_mul(float* result, const float* matrix_a,
    const float* matrix_b, int rows_a, int cols_a, int rows_b, int cols_b){
    // product would be A * B^T
    if (cols_a != cols_b){
        fprintf(stderr, "%s: dimension mismatch\n", __func__);
        return;
    }
    xlns32_float sum;

    for (int i = 0; i < rows_a; ++i){
        
        for (int j = 0; j < rows_b; ++j){
            sum = 0;
            for (int k = 0; k < cols_a; ++k){
                xlns32_float a, b;
                a = matrix_a[i * cols_a + k];
                b = matrix_b[j * cols_b + k];
                sum += a * b;
            }
            result[i * rows_b + j] = xlns32_2float(sum);
        }
    }
}


int main(){
    const int rows_a = 4, cols_a = 2;
    const int rows_b = 3, cols_b = 2;

    const float matrix_A[rows_a * cols_a] = {
        2, 8,
        5, 1,
        4, 2,
        8, 6,
    };

    const float matrix_B[rows_b * cols_b] = {
        10, 5,
        9, 9,
        5, 4,
    };



    float result[rows_a * rows_b] = {0};

    mat_mul(result, matrix_A, matrix_B, rows_a, cols_a, rows_b, cols_b);


    for (int i = 0; i < rows_a; ++i){
        for (int j = 0; j < rows_b; ++j){
            printf(" %.3f ", result[i * rows_b + j]);
        }
        printf("\n");
    }

    return 0;
}