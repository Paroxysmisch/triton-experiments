cpp
#include <triton/kernel.h>

__device__ float softmax(float x) {
    return tl::math::exp2(x);
}

__global__ void _fwd_kernel_aligned(
    float* Q, float* K, float* V, float* B0, float* Out,
    int BLOCK_M, int BLOCK_N, int BLOCK_DMODEL,
    float sm_scale) {

    // Calculate block and thread indices
    int bm = blockIdx.x % BLOCK_M;
    int bn = blockIdx.y % BLOCK_N;
    int bd = blockIdx.z % BLOCK_DMODEL;
    int tm = threadIdx.x;
    int tn = threadIdx.y;

    // Load Q, K, V, and B0 into shared memory
    extern __shared__ float shared[];
    float* sQ = shared;
    float* sK = &shared[BLOCK_DMODEL * BLOCK_M * BLOCK_N];
    float* sV = &shared[2 * BLOCK_DMODEL * BLOCK_M * BLOCK_N];
    float* sB0 = &shared[3 * BLOCK_DMODEL * BLOCK_M * BLOCK_N];

    // Load data into shared memory
    sQ[tn * BLOCK_M + tm] = Q[bd * BLOCK_DMODEL * BLOCK_M * BLOCK_N + bn * BLOCK_M + tm];
    sK[tn * BLOCK_M + tm] = K[bd * BLOCK_DMODEL * BLOCK_M * BLOCK_N + bn * BLOCK_M + tm];
    sV[tn * BLOCK_M + tm] = V[bd * BLOCK_DMODEL * BLOCK_M * BLOCK_N + bn * BLOCK_M + tm];
    sB0[tn * BLOCK_M + tm] = B0[bd * BLOCK_DMODEL * BLOCK_M * BLOCK_N + bn * BLOCK_M + tm];

    // Synchronize to make sure all data is loaded
    __syncthreads();

    // Perform dot product and bias addition
    float dot_product = 0.0f;
    for (int i = 0; i < BLOCK_M; i++) {
        dot_product += sQ[tn * BLOCK_M + i] * sK[i * BLOCK_N + tm];
    }
    dot_product += sB0[tn * BLOCK_M + tm];

    // Apply softmax
    float exp_dot_product = softmax(dot_product * sm_scale);

    // Store result
    Out[bd * BLOCK_DMODEL * BLOCK_M * BLOCK_N + bn * BLOCK_M + tm] = exp_dot_product;
}

void _attention_rel_h_rel_w_kernel_aligned_device(
    float* Q, float* K, float* V, float* B0, float* Out,
    int BLOCK_M, int BLOCK_N, int BLOCK_DMODEL,
    float sm_scale, int num_warps, int num_stages) {

    // Validate input tensor shapes and types
    // ...

    // Calculate necessary constants
    // ...

    // Set up computation grid
    dim3 grid(Q->shape[0], Q->shape[1], Q->shape[2]);
    dim3 block(BLOCK_M, BLOCK_N);

    // Launch kernel
    _fwd_kernel_aligned<<<grid, block, 4 * BLOCK_DMODEL * BLOCK_M * BLOCK_N * sizeof(float)>>>(
        Q, K, V, B0, Out, BLOCK_M, BLOCK_N, BLOCK_DMODEL, sm_scale);
}
