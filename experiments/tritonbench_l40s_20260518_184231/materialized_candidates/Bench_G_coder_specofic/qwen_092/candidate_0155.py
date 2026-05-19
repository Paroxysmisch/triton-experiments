cpp
#include <triton/triton.h>

__global__ void chunk_linear_attn_fwd_kernel_h(
    float* __restrict__ h,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ initial_state,
    int T,
    int K,
    int V,
    int BT,
    int BK,
    int BV) {
    // Triton block pointers
    const int block_size = 32;
    const int block_x = blockIdx.x * block_size;
    const int block_y = blockIdx.y * block_size;
    const int block_z = blockIdx.z * block_size;

    // Thread indices within the block
    const int thread_x = threadIdx.x;
    const int thread_y = threadIdx.y;
    const int thread_z = threadIdx.z;

    // Compute global indices
    const int t = block_x + thread_x;
    const int k_idx = block_y + thread_y;
    const int v_idx = block_z + thread_z;

    // Initialize h if there is an initial state
    if (initial_state != nullptr) {
        h[t * K * V + k_idx * V + v_idx] = initial_state[t * K * V + k_idx * V + v_idx];
    } else {
        h[t * K * V + k_idx * V + v_idx] = 0.0f;
    }

    // Iterate over time steps
    for (int step = 1; step < T; ++step) {
        // Load k and v for the current step
        const float* k_ptr = k + step * K * V;
        const float* v_ptr = v + step * K * V;

        // Compute dot product and update h
        for (int i = 0; i < V; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < K; ++j) {
                sum += k_ptr[k_idx * K + j] * v_ptr[j * V + i];
            }
            h[t * K * V + k_idx * V + v_idx] += sum;
        }
    }
}
