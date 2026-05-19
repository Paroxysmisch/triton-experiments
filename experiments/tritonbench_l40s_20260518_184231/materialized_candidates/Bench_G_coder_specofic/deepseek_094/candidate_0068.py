c++
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <triton/kernel_interface.h>

#define BLOCK 8
#define BLOCK_N 128

__device__ void _multinomial_sampling_kernel(
    const float* __restrict__ scores,
    unsigned int* __restrict__ indices,
    unsigned int batch_size,
    unsigned int num_tokens,
    curandStatePhilox4_32_10_t* states,
    unsigned int seed) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;

    curand_init(seed, tid, 0, &states[tid]);

    unsigned int offset = bid * num_tokens;
    float sum = 0.0f;

    if (tid < num_tokens) {
        sum = scores[offset + tid];
    }

    __shared__ float cumulative_scores[BLOCK_N];
    __shared__ float block_sum;

    for (unsigned int stride = 1; stride < BLOCK_N; stride *= 2) {
        __syncthreads();
        if (tid % (2 * stride) == 0) {
            unsigned int idx = tid + stride;
            if (idx < BLOCK_N) {
                cumulative_scores[idx] += cumulative_scores[tid];
            }
        }
        __syncthreads();
        if (tid < BLOCK_N) {
            cumulative_scores[tid] += sum;
        }
    }

    __syncthreads();

    if (tid == 0) {
        block_sum = cumulative_scores[BLOCK_N - 1];
    }

    __syncthreads();

    if (tid < BLOCK_N) {
        cumulative_scores[tid] /= block_sum;
    }

    __syncthreads();

    float u = curand_uniform(&states[tid]) * block_sum;
    unsigned int i = 0;

    while (i < BLOCK_N && cumulative_scores[i] < u) {
        i++;
    }

    if (tid < num_tokens) {
        indices[offset + tid] = i;
    }
}

extern "C" __triton__
void multinomial_sampling(
    const float* __restrict__ scores,
    unsigned int* __restrict__ indices,
    unsigned int batch_size,
    unsigned int num_tokens,
    unsigned int seed) {
    dim3 grid(batch_size, 1, 1);
    dim3 block(BLOCK_N, 1, 1);

    curandStatePhilox4_32_10_t* states;
    cudaMalloc(&states, BLOCK_N * sizeof(curandStatePhilox4_32_10_t));

    _multinomial_sampling_kernel<<<grid, block>>>(scores, indices, batch_size, num_tokens, states, seed);

    cudaFree(states);
}
