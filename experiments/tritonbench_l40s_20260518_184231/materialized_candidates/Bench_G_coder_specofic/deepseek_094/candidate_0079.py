cpp
#include <cuda_runtime.h>
#include <triton/api.h>

#define BLOCK_M 16
#define BLOCK_N 16
#define GROUP_M 8

__device__ void _quantize_global_transpose(int8_t* B, float* A, int M, int N, int stride_am, int stride_an, int stride_bm, int stride_bn, float absmax, float absmax_inv) {
    int m = blockIdx.y * BLOCK_M + threadIdx.y;
    int n = blockIdx.x * BLOCK_N + threadIdx.x;

    if (m < M && n < N) {
        int idx_a = m * stride_am + n * stride_an;
        int idx_b = n * stride_bm + m * stride_bn;

        float val = A[idx_a];
        int8_t quantized_val = (int8_t) (val * absmax_inv);

        B[idx_b] = quantized_val;
    }
}

void quantize_global_transpose(float* A, int M, int N, int8_t* B, float absmax, float absmax_inv) {
    int stride_am = GROUP_M;
    int stride_an = N;
    int stride_bm = N;
    int stride_bn = 1;

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(BLOCK_N, BLOCK_M);

    triton::api::launch('_quantize_global_transpose', grid, block, 0, 0, B, A, M, N, stride_am, stride_an, stride_bm, stride_bn, absmax, absmax_inv);
}
