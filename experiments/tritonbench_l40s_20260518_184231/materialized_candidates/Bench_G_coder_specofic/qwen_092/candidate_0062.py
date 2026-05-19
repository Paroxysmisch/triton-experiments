cpp
#include <cuda_runtime.h>
#include <triton/core/triton.h>
#include <triton/core/triton_types.h>

#define BLOCK_N 32
#define BLOCK_NN 64
#define BLOCK_DMODEL 128

__global__ void embedding_kernel(
    const float* __restrict__ weights,
    const int* __restrict__ token_ids,
    float* __restrict__ out,
    const int seq_len,
    const int vocab_size,
    const int d_model,
    const int* __restrict__ masks) {
    
    __shared__ float shared_weights[BLOCK_DMODEL][BLOCK_NN];
    __shared__ int shared_masks[BLOCK_NN];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int seq_idx = bid * BLOCK_N + tid;

    if (seq_idx >= seq_len) {
        return;
    }

    int token_id = token_ids[seq_idx];
    int mask = masks[seq_idx];

    // Load weights and masks into shared memory
    if (tid < BLOCK_DMODEL) {
        for (int j = 0; j < BLOCK_NN; j++) {
            shared_weights[tid][j] = weights[token_id * d_model + j];
        }
    }
    if (tid < BLOCK_NN) {
        shared_masks[tid] = mask;
    }

    __syncthreads();

    // Compute the output vector for the current sequence position
    for (int j = 0; j < BLOCK_DMODEL; j++) {
        out[seq_idx * d_model + j] = 0.0f;
        for (int k = 0; k < BLOCK_NN; k++) {
            if (shared_masks[k] == 1) {
                out[seq_idx * d_model + j] += shared_weights[j][k];
            }
        }
    }
}

__global__ void embedding_wrapper(
    const float* __restrict__ weights,
    const int* __restrict__ token_ids,
    float* __restrict__ out,
    const int seq_len,
    const int vocab_size,
    const int d_model,
    const int* __restrict__ masks) {
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int seq_idx = bid * BLOCK_N + tid;

    if (seq_idx >= seq_len) {
        return;
    }

    int token_id = token_ids[seq_idx];
    int mask = masks[seq_idx];

    int offset = token_id * d_model;
    int out_offset = seq_idx * d_model;

    // Load weights and masks into shared memory
    __shared__ float shared_weights[BLOCK_DMODEL][BLOCK_NN];
    __shared__ int shared_masks[BLOCK_NN];

    for (int j = 0; j < BLOCK_DMODEL; j++) {
        for (int k = 0; k < BLOCK_NN; k++) {
            shared_weights[j][k] = weights[offset + j * BLOCK_NN + k];
        }
    }
    for (int k = 0; k < BLOCK_NN; k++) {
        shared_masks[k] = mask;
    }

    __syncthreads();

    // Compute the output vector for the current sequence position
    for (int j = 0; j < BLOCK_DMODEL; j++) {
        out[out_offset + j] = 0.0f;
        for (int k = 0; k < BLOCK_NN; k++) {
            if (shared_masks[k] == 1) {
                out[out_offset + j] += shared_weights[j][k];
            }
        }
    }
}
