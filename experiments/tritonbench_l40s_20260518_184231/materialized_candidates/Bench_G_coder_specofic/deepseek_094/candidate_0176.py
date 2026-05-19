cpp
#include <ampere_smem.h>
#include <ampere_warp.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <triton/ir.h>
#include <triton/smem.h>
#include <triton/warp.h>

using namespace triton;

// Define the kernel
void _fwd_kernel_token_softmax(
    float* input,
    float* output,
    const int batch_size,
    const int sequence_length,
    const int num_heads,
    const int head_size,
    const int* sequence_lengths)
{
    // Define the grid and block dimensions
    const int grid_size = batch_size * num_heads;
    const int block_size = sequence_length;

    // Define the shared memory
    __shared__ float smem[head_size];

    // Define the dynamic shared memory
    float* dynamic_smem = smem + head_size;

    // Define the grid and block indices
    const int grid_idx = blockIdx.x;
    const int block_idx = threadIdx.x;

    // Define the batch and head indices
    const int batch_idx = grid_idx / num_heads;
    const int head_idx = grid_idx % num_heads;

    // Define the sequence index
    const int sequence_idx = block_idx;

    // Load the appropriate segment of logits for each token sequence
    float logit = input[((batch_idx * num_heads + head_idx) * sequence_length + sequence_idx)];

    // Apply a numerically stable softmax by subtracting the maximum logit value from each element
    float max_logit = triton::warp::reduce_max(logit);
    logit -= max_logit;

    // Normalize by the total exponentiated sum
    float exp_logit = triton::warp::exp(logit);
    float sum_exp_logit = triton::warp::reduce_sum(exp_logit);
    output[((batch_idx * num_heads + head_idx) * sequence_length + sequence_idx)] = exp_logit / sum_exp_logit;

    // Handle padded sequences
    if (sequence_idx >= sequence_lengths[batch_idx]) {
        output[((batch_idx * num_heads + head_idx) * sequence_length + sequence_idx)] = -INFINITY;
    }
}

// Define the wrapper function
void token_softmax_fwd(
    float* input,
    float* output,
    const int batch_size,
    const int sequence_length,
    const int num_heads,
    const int head_size,
    const int* sequence_lengths)
{
    // Define the grid and block dimensions
    dim3 grid_dim(batch_size * num_heads);
    dim3 block_dim(sequence_length);

    // Launch the kernel
    _fwd_kernel_token_softmax<<<grid_dim, block_dim>>>(
        input,
        output,
        batch_size,
        sequence_length,
        num_heads,
        head_size,
        sequence_lengths);
}
