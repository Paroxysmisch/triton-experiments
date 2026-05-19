cpp
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void decoding_fused_rotary_embedding_kernel(float* q, float* k_cache, float* v_cache, int q_total_tokens, int q_head_num, int kv_length, int KV_GROUP_NUM, bool use_new_kcache_layout) {
    // Kernel implementation here
}

extern "C" {
    void decoding_fused_rotary_embedding(float* q, float* k_cache, float* v_cache, int q_total_tokens, int q_head_num, int kv_length, int KV_GROUP_NUM, bool use_new_kcache_layout) {
        dim3 threadsPerBlock(128);
        dim3 numBlocks((q_total_tokens + threadsPerBlock.x - 1) / threadsPerBlock.x);
        decoding_fused_rotary_embedding_kernel<<<numBlocks, threadsPerBlock>>>(q, k_cache, v_cache, q_total_tokens, q_head_num, kv_length, KV_GROUP_NUM, use_new_kcache_layout);
    }
}
