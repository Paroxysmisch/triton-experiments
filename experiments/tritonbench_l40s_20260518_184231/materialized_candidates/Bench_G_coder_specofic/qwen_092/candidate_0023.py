cpp
#include <triton/core/triton.h>
#include <cmath>

// Constants
const int Q_HEAD_NUM = 8;
const int HEAD_DIM = 64;

// Kernel for rotary_embedding when k_cache is None
__global__ void rotary_embedding_kernel(
    float* q, float* k, float* cos, float* sin,
    int batch_size, int seq_len, int head_dim, int num_warps) {
    
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int head_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (q_idx >= batch_size * seq_len * head_dim || k_idx >= batch_size * seq_len * head_dim) {
        return;
    }

    int q_offset = q_idx * head_dim;
    int k_offset = k_idx * head_dim;
    int cos_offset = head_idx * head_dim;
    int sin_offset = head_idx * head_dim;

    float* q0 = &q[q_offset];
    float* q1 = &q[q_offset + head_dim / 2];
    float* k0 = &k[k_offset];
    float* k1 = &k[k_offset + head_dim / 2];
    float* cos_val = &cos[cos_offset];
    float* sin_val = &sin[sin_offset];

    float out_q0 = q0[0] * cos_val[0] - q1[0] * sin_val[0];
    float out_q1 = q0[0] * sin_val[0] + q1[0] * cos_val[0];

    q0[0] = out_q0;
    q1[0] = out_q1;

    if (k0 != nullptr) {
        float out_k0 = k0[0] * cos_val[0] - k1[0] * sin_val[0];
        float out_k1 = k0[0] * sin_val[0] + k1[0] * cos_val[0];

        k0[0] = out_k0;
        k1[0] = out_k1;
    }
}

// Kernel for rotary_embedding when k_cache is provided
__global__ void fused_rotary_embedding_kernel_v2(
    float* q, float* k, float* cos, float* sin, float* k_cache, float* block_tables, float* kv_lengths,
    int batch_size, int seq_len, int head_dim, int num_warps) {
    
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int head_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (q_idx >= batch_size * seq_len * head_dim || k_idx >= batch_size * seq_len * head_dim) {
        return;
    }

    int q_offset = q_idx * head_dim;
    int k_offset = k_idx * head_dim;
    int cos_offset = head_idx * head_dim;
    int sin_offset = head_idx * head_dim;
    int k_cache_offset = head_idx * seq_len * head_dim;

    float* q0 = &q[q_offset];
    float* q1 = &q[q_offset + head_dim / 2];
    float* k0 = &k[k_offset];
    float* k1 = &k[k_offset + head_dim / 2];
    float* cos_val = &cos[cos_offset];
    float* sin_val = &sin[sin_offset];
    float* k_cache0 = &k_cache[k_cache_offset];
    float* k_cache1 = &k_cache[k_cache_offset + head_dim / 2];

    float out_q0 = q0[0] * cos_val[0] - q1[0] * sin_val[0];
    float out_q1 = q0[0] * sin_val[0] + q1[0] * cos_val[0];

    q0[0] = out_q0;
    q1[0] = out_q1;

    if (k0 != nullptr) {
        float out_k0 = k0[0] * cos_val[0] - k1[0] * sin_val[0];
        float out_k1 = k0[0] * sin_val[0] + k1[0] * cos_val[0];

        k0[0] = out_k0;
        k1[0] = out_k1;

        int block_idx = block_tables[q_idx];
        int kv_length = kv_lengths[block_idx];

        if (q_idx >= kv_length) {
            k_cache0[0] = out_k0;
            k_cache1[0] = out_k1;
        }
    }
}
