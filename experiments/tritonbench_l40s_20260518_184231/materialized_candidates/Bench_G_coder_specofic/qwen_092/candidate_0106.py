triton
// iv_dependent_matmul_kernel.triton

#include <triton/triton.h>

// Define the kernel function
@triton.kernel
void iv_dependent_matmul_kernel(
    float* a, float* b, float* c,
    int32_t M, int32_t N, int32_t K,
    int32_t stride_a_M, int32_t stride_a_K,
    int32_t stride_b_K, int32_t stride_b_N,
    int32_t stride_c_M, int32_t stride_c_N,
    int32_t BLOCK_SIZE_M, int32_t BLOCK_SIZE_N, int32_t BLOCK_SIZE_K,
    const char* type
) {
    // Get the program ID and thread ID
    int32_t pid = get_program_id(0);
    int32_t tid = get_thread_id(0);

    // Calculate block indices
    int32_t block_row = pid / (BLOCK_SIZE_N * BLOCK_SIZE_K);
    int32_t block_col = (pid % (BLOCK_SIZE_N * BLOCK_SIZE_K)) / BLOCK_SIZE_K;
    int32_t block_k = (pid % (BLOCK_SIZE_N * BLOCK_SIZE_K)) % BLOCK_SIZE_K;

    // Calculate block offsets
    int32_t row_offset = block_row * BLOCK_SIZE_M;
    int32_t col_offset = block_col * BLOCK_SIZE_N;
    int32_t k_offset = block_k * BLOCK_SIZE_K;

    // Declare shared memory for block matrices
    __shared__ float block_a[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float block_b[BLOCK_SIZE_K][BLOCK_SIZE_N];

    // Initialize block matrices to zero
    for (int32_t i = 0; i < BLOCK_SIZE_M; i++) {
        for (int32_t j = 0; j < BLOCK_SIZE_K; j++) {
            block_a[i][j] = 0.0f;
        }
    }
    for (int32_t i = 0; i < BLOCK_SIZE_K; i++) {
        for (int32_t j = 0; j < BLOCK_SIZE_N; j++) {
            block_b[i][j] = 0.0f;
        }
    }

    // Synchronize to ensure all shared memory is initialized
    __syncthreads();

    // Load block matrices into shared memory
    if (tid < BLOCK_SIZE_M) {
        for (int32_t k = 0; k < BLOCK_SIZE_K; k++) {
            block_a[tid][k] = a[row_offset + tid * stride_a_M + k * stride_a_K];
        }
    }
    if (tid < BLOCK_SIZE_K) {
        for (int32_t n = 0; n < BLOCK_SIZE_N; n++) {
            block_b[tid][n] = b[k_offset + tid * stride_b_K + n * stride_b_N];
        }
    }

    // Synchronize to ensure all shared memory is loaded
    __syncthreads();

    // Perform matrix multiplication
    float acc[BLOCK_SIZE_N];
    for (int32_t i = 0; i < BLOCK_SIZE_N; i++) {
        acc[i] = 0.0f;
        for (int32_t k = 0; k < BLOCK_SIZE_K; k++) {
            acc[i] += block_a[tid][k] * block_b[k][i];
        }
    }

    // Store the result in global memory
    int32_t result_row = row_offset + tid;
    int32_t result_col = col_offset + get_thread_id(1);
    if (tid < BLOCK_SIZE_N) {
        c[result_row * stride_c_M + result_col * stride_c_N] = acc[tid];
    }
}
