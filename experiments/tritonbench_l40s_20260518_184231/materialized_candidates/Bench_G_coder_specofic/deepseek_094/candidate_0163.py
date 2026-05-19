cpp
#include <cuda_runtime.h>
#include <triton/kernel_definitions.h>

// Define the kernel function
__device__
void matmul_kernel(float* A, float* B, float* C, int M, int N, int K, const char* activation) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        // Apply activation function if needed
        if (strcmp(activation, "leaky_relu") == 0) {
            C[row * N + col] = fmaxf(0.01f * sum, sum);
        } else {
            C[row * N + col] = sum;
        }
    }
}

// Define the wrapper function
void matmul(float* A, float* B, float* C, int M, int N, int K, const char* activation) {
    dim3 gridDim(ceil(N / BLOCK_SIZE_N), ceil(M / BLOCK_SIZE_M), 1);
    dim3 blockDim(BLOCK_SIZE_N, BLOCK_SIZE_M, 1);

    // Initialize C to zero
    cudaMemset(C, 0, M * N * sizeof(float));

    // Invoke the kernel
    matmul_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K, activation);
}
