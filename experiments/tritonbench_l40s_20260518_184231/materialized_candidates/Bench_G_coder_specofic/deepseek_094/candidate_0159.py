cpp
#include <cuda.h>
#include <triton/kernel.h>

__device__ void nested3(int i, int j, int k, int stride_m, int stride_n,
                        float* in_ptr, float* out_ptr) {
    int a_offset = i * stride_m + j * stride_n + k;
    int c_offset = i * stride_m + j * stride_n + k;

    float a_val = tl::load(in_ptr + a_offset);
    float c_val = a_val;

    tl::store(out_ptr + c_offset, c_val);
}

void wrapper_nested3(int n_rows, int n_cols, float* x, float* output) {
    dim3 grid(n_cols / 4, n_rows);
    dim3 block(4, 4);

    int stride_m = n_cols * n_rows;
    int stride_n = n_cols;

    nested3<<<grid, block>>>(n_rows, n_cols, stride_m, stride_n, x, output);

    cudaDeviceSynchronize();

    // Print the output tensor
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            printf("%f ", output[i * n_cols + j]);
        }
        printf("\n");
    }
}
