triton
#include <triton/triton.h>

#define BLOCK_SIZE_M 32
#define BLOCK_SIZE_N 32
#define BLOCK_SIZE_K 32

// Helper function to compute the square of a value
__device__ float square(float x) {
    return x * x;
}

// Triton kernel for RMS normalization and matrix multiplication
__global__ void rms_matmul_rbe(
    float* __restrict__ x,        // Input tensor
    float* __restrict__ w,        // Transposed weight matrix
    float* __restrict__ rms_w,    // RMS weight
    float* __restrict__ output,   // Output tensor
    int batch_size,               // Batch size
    int num_heads,                // Number of attention heads
    int head_size,                // Head size
    float theta,                  // Rotary embedding angle
    bool use_rotary_embedding     // Whether to use rotary embeddings
) {
    // Indices
    int batch = blockIdx.x / (BLOCK_SIZE_M * BLOCK_SIZE_N);
    int m = blockIdx.x % (BLOCK_SIZE_M * BLOCK_SIZE_N) / BLOCK_SIZE_N;
    int n = blockIdx.x % BLOCK_SIZE_N;

    // Shared memory for input and weight tiles
    __shared__ float x_tile[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float w_tile[BLOCK_SIZE_K][BLOCK_SIZE_N];

    // Shared memory for RMS normalization
    __shared__ float sum_squares[BLOCK_SIZE_M];
    __shared__ float mean[BLOCK_SIZE_M];
    __shared__ float rsqrt[BLOCK_SIZE_M];

    // Load input tile into shared memory
    int x_idx = batch * num_heads * head_size * batch_size + m * head_size * batch_size + n * batch_size;
    for (int k = 0; k < BLOCK_SIZE_K; k++) {
        x_tile[m][k] = x[x_idx + k];
    }

    // Load weight tile into shared memory
    int w_idx = batch * num_heads * head_size * batch_size + n * head_size * batch_size;
    for (int k = 0; k < BLOCK_SIZE_K; k++) {
        w_tile[k][n] = w[w_idx + k];
    }

    // Compute sum of squares for RMS normalization
    float sum_square = 0.0f;
    for (int k = 0; k < BLOCK_SIZE_K; k++) {
        sum_square += square(x_tile[m][k]);
    }
    sum_squares[m] = sum_square;

    // Barrier to ensure all threads have computed their sum of squares
    __syncthreads();

    // Reduce sum of squares across threads in the block
    for (int s = BLOCK_SIZE_M / 2; s > 0; s /= 2) {
        if (m < s) {
            sum_squares[m] += sum_squares[m + s];
        }
        __syncthreads();
    }

    // Compute mean and reciprocal square root for RMS normalization
    if (m == 0) {
        mean[0] = sum_squares[0] / (batch_size * head_size);
        rsqrt[0] = 1.0f / sqrt(mean[0] + 1e-6f);
    }
    __syncthreads();

    // Compute output element
    float result = 0.0f;
    for (int k = 0; k < BLOCK_SIZE_K; k++) {
        result += x_tile[m][k] * w_tile[k][n];
    }

    // Apply RMS normalization
    result *= rsqrt[0];

    // Apply rotary embeddings if specified
    if (use_rotary_embedding) {
        float theta_m = theta * (m + 0.5f);
        float cos_theta = cos(theta_m);
        float sin_theta = sin(theta_m);
        result = cos_theta * result - sin_theta * result;
    }

    // Store output element
    int output_idx = batch * num_heads * head_size * batch_size + m * head_size * batch_size + n * batch_size;
    output[output_idx] = result;
}
