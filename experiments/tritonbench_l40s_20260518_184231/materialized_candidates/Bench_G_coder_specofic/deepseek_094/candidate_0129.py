c++
#include <cuda_runtime.h>
#include <triton/kernel_definitions.h>

#define WARPS_PER_BLOCK 8

__device__
void apply_penalty(
    float* logits,
    const float* penalties,
    const int* p_cumsum_seq_len,
    const int* p_token_ids,
    const int* p_token_counts,
    int seq_len,
    int vocab_size,
    int cur_batch,
    int batch_size,
    int stride
) {
    int token_id = threadIdx.x % vocab_size;
    int warp_id = threadIdx.x / vocab_size;
    int batch_offset = cur_batch * seq_len;
    int warp_offset = warp_id * vocab_size;

    int start = p_cumsum_seq_len[cur_batch];
    int end = p_cumsum_seq_len[cur_batch + 1];

    for (int i = start + warp_offset + token_id; i < end; i += WARPS_PER_BLOCK * vocab_size) {
        int count = p_token_counts[i];
        int token_id = p_token_ids[i];
        float penalty = penalties[token_id];

        float logit = logits[i * stride + batch_offset];
        logit *= (1.0 - penalty);
        logits[i * stride + batch_offset] = logit;
    }
}

__global__
void _fwd_kernel_apply_penalty(
    float* logits,
    const float* penalties,
    const int* p_cumsum_seq_len,
    const int* p_token_ids,
    const int* p_token_counts,
    int seq_len,
    int vocab_size,
    int batch_size,
    int stride
) {
    int cur_batch = blockIdx.x;
    if (cur_batch < batch_size) {
        apply_penalty(
            logits,
            penalties,
            p_cumsum_seq_len,
            p_token_ids,
            p_token_counts,
            seq_len,
            vocab_size,
            cur_batch,
            batch_size,
            stride
        );
    }
}

void apply_penalty(
    float* logits,
    const float* penalties,
    const int* p_cumsum_seq_len,
    const int* p_token_ids,
    const int* p_token_counts,
    int seq_len,
    int vocab_size,
    int batch_size,
    int stride
) {
    cudaMemset(logits, 0, batch_size * seq_len * sizeof(float));
    _fwd_kernel_apply_penalty<<<batch_size, WARPS_PER_BLOCK * vocab_size>>>(
        logits,
        penalties,
        p_cumsum_seq_len,
        p_token_ids,
        p_token_counts,
        seq_len,
        vocab_size,
        batch_size,
        stride
    );
}
