cpp
#include <triton/api.h>

#define BLOCK_SIZE_M 16
#define BLOCK_SIZE_N 16
#define BLOCK_SIZE_K 16

#define ACTIVATION 0.01f

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int A_offset = wA * K * by;
    int B_offset = K * bx;
    int C_offset = wC * N * by;

    float Cvalue = 0.0f;

    for (int k = 0; k < K; ++k) {
        float A_element = A[A_offset + wA * ty + k];
        float B_element = B[B_offset + wB * k + tx];
        Cvalue += A_element * B_element;
    }

    int C_index = C_offset + wC * ty + tx;
    if (ACTIVATION > 0) {
        C[C_index] = max(Cvalue, 0.0f) + ACTIVATION * min(Cvalue, 0.0f);
    } else {
        C[C_index] = Cvalue;
    }
}
