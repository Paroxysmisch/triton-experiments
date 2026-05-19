c++
#include <cuda_runtime.h>
#include <stdio.h>
#include <cooperative_groups.h>
#include <triton/language.h>

namespace cg = cooperative_groups;

__global__ void rotary_kernel(float* OUT, float* X, float* COS, float* SIN, int* SEQLENS, int* CU_SEQLENS, int interleaved, int conjugate, int batch_size, int num_heads, int seq_len) {
    // Setup cooperative groups
    cg::grid_group grid = cg::this_grid();
    cg::thread_group threads = cg::this_thread_block();

    // Determine the batch, head and sequence index
    int pid_batch = blockIdx.z;
    int pid_head = blockIdx.y;
    int pid_m = threadIdx.x;

    // Load sequence length
    int seqlen = SEQLENS[pid_batch * num_heads + pid_head];

    // Compute the rotary position
    int rotary_pos = seqlen + pid_m;

    // Load the cosine and sine values
    float cos_val = COS[rotary_pos];
    float sin_val = SIN[rotary_pos];

    // Load the input value
    float x_val = X[pid_batch * num_heads * seq_len + pid_head * seq_len + pid_m];

    // Apply the rotary position encoding
    float out_val = x_val * cos_val - X[pid_batch * num_heads * seq_len + pid_head * seq_len + pid_m] * sin_val;

    // Store the output value
    OUT[pid_batch * num_heads * seq_len + pid_head * seq_len + pid_m] = out_val;
}

void apply_rotary(float* x, float* cos, float* sin, int* seqlens, int* cu_seqlens, int interleaved, int conjugate, int batch_size, int num_heads, int seq_len) {
    // Determine the grid and block sizes
    dim3 gridSize(1, num_heads, batch_size);
    dim3 blockSize(seq_len, 1, 1);

    // Copy the input tensor to the output tensor
    float* out;
    cudaMalloc(&out, batch_size * num_heads * seq_len * sizeof(float));
    cudaMemcpy(out, x, batch_size * num_heads * seq_len * sizeof(float), cudaMemcpyDeviceToDevice);

    // Call the rotary kernel
    rotary_kernel<<<gridSize, blockSize>>>(out, x, cos, sin, seqlens, cu_seqlens, interleaved, conjugate, batch_size, num_heads, seq_len);

    // Copy the output tensor back to the host
    cudaMemcpy(x, out, batch_size * num_heads * seq_len * sizeof(float), cudaMemcpyDeviceToDevice);

    // Free the output tensor
    cudaFree(out);
}
