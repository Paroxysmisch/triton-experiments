triton
#include <triton/core.h>
#include <triton/language.h>

// Triton kernel function for applying rotary embedding transformation
__triton__ __device__ void decoding_fused_rotary_embedding_kernel(
    float* q, float* k, float* v, float* k_cache, float* v_cache,
    const float* sin_lookup, const float* cos_lookup,
    const int q_head_num, const int q_total_tokens, const int head_dim,
    const int KV_GROUP_NUM, const int kv_length, const int kv_stride,
    const int q_stride, const int k_stride, const int v_stride,
    const int k_cache_stride, const int v_cache_stride,
    const int sin_lookup_stride, const int cos_lookup_stride,
    const int use_new_kcache_layout) {
    // Get the current program ID
    int program_id = triton::program_id(0);
    int head_id = program_id / q_total_tokens;
    int token_id = program_id % q_total_tokens;

    // Calculate the indices for q, k, v, k_cache, and v_cache
    int q_idx = head_id * q_stride + token_id;
    int k_idx = head_id * k_stride + token_id;
    int v_idx = head_id * v_stride + token_id;
    int k_cache_idx = head_id * k_cache_stride + token_id;
    int v_cache_idx = head_id * v_cache_stride + token_id;

    // Calculate the indices for sin and cos lookup tables
    int sin_idx = head_id * head_dim / 2;
    int cos_idx = head_id * head_dim / 2;

    // Apply rotary embedding transformation to q
    for (int i = 0; i < head_dim / 2; ++i) {
        float sin_val = sin_lookup[sin_idx + i];
        float cos_val = cos_lookup[cos_idx + i];
        float q_i = q[q_idx + i];
        float q_i_plus_half = q[q_idx + i + head_dim / 2];

        q[q_idx + i] = q_i * cos_val - q_i_plus_half * sin_val;
        q[q_idx + i + head_dim / 2] = q_i * sin_val + q_i_plus_half * cos_val;
    }

    // Update k_cache and v_cache if needed
    if (use_new_kcache_layout) {
        if (token_id < kv_length) {
            k_cache[k_cache_idx] = k[k_idx];
            v_cache[v_cache_idx] = v[v_idx];
        }
    } else {
        if (token_id < kv_length) {
            k_cache[k_cache_idx] = k[k_idx];
            v_cache[v_cache_idx] = v[v_idx];
        }
    }
}

// Triton wrapper function for setting up kernel execution
__triton__ __device__ void decoding_fused_rotary_embedding(
    float* q, float* k, float* v, float* k_cache, float* v_cache,
    const float* sin_lookup, const float* cos_lookup,
    const int q_head_num, const int q_total_tokens, const int head_dim,
    const int KV_GROUP_NUM, const int kv_length, const int kv_stride,
    const int q_stride, const int k_stride, const int v_stride,
    const int k_cache_stride, const int v_cache_stride,
    const int sin_lookup_stride, const int cos_lookup_stride,
    const int use_new_kcache_layout) {
    // Define the grid size
    triton::grid1d(q_head_num * q_total_tokens);

    // Define the block size
    int warp_size = triton::warp_size();
    int head_per_block = 32 / warp_size; // Adjust based on head_dim
    int block_size = head_per_block * warp_size;

    // Launch the kernel
    decoding_fused_rotary_embedding_kernel<<<triton::num_warps(block_size), block_size>>>(
        q, k, v, k_cache, v_cache, sin_lookup, cos_lookup,
        q_head_num, q_total_tokens, head_dim,
        KV_GROUP_NUM, kv_length, kv_stride,
        q_stride, k_stride, v_stride,
        k_cache_stride, v_cache_stride,
        sin_lookup_stride, cos_lookup_stride,
        use_new_kcache_layout);
}
