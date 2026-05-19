cpp
#include <cuda.h>
#include <cuda_runtime.h>
#include <triton/kernel_config.h>

#define TILE_N 1024

__device__ float reduceMax(float* data, int len) {
    float max_val = -1e10f;
    for (int i = 0; i < len; ++i) {
        max_val = max(max_val, data[i]);
    }
    return max_val;
}

__device__ float reduceSum(float* data, int len) {
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        sum += expf(data[i]);
    }
    return sum;
}

__global__ void softmax_kernel_online_v2(float* input, float* output, int M, int N, int TILE_N) {
    extern __shared__ float tile[];

    int row = blockIdx.y * TILE_N + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    float max_val = -1e10f;
    float sum = 0.0f;

    if (row < M && col < N) {
        max_val = reduceMax(input + row * N + col, N);
        sum = reduceSum(input + row * N + col, N);

        for (int i = 0; i < N; ++i) {
            output[row * N + i] = expf(input[row * N + i] - max_val) / sum;
        }
    }
}

void softmax(float* input, float* output, int M, int N) {
    dim3 threadsPerBlock(TILE_N, TILE_N);
    dim3 numBlocks((N + TILE_N - 1) / TILE_N, (M + TILE_N - 1) / TILE_N);

    softmax_kernel_online_v2<<<numBlocks, threadsPerBlock>>>(input, output, M, N, TILE_N);
}
