triton
#include <triton/triton.h>

#define BLOCK_SIZE 256

__global__ void diag_ssm_forward_kernel(
    float* s_ptr,   // Input initial states
    float* x_ptr,   // Input sequence data
    float* y_ptr,   // Output sequence data
    float* Lambda,  // Transformation matrix
    int length,     // Length of the sequence
    int batch_size, // Batch size
    int dim         // Dimension of the states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int t = idx / (batch_size * dim);
    int b = (idx / dim) % batch_size;
    int d = idx % dim;

    if (t < length && b < batch_size && d < dim) {
        int s_idx = b * dim + d;
        int x_idx = t * batch_size * dim + s_idx;
        int y_idx = t * batch_size * dim + s_idx;

        float s = s_ptr[s_idx];
        float x = x_ptr[x_idx];
        float lambda = Lambda[d];

        y_ptr[y_idx] = s * lambda + x;
    }
}
