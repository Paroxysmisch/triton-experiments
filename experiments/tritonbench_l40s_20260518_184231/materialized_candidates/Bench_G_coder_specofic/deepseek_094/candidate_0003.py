c++
__device__ void _fwd_kernel(
    float* __restrict__ output,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const int batch,
    const int head,
    const int seq_len,
    const int dim_per_head,
    const float scale) {

    const int q_idx = blockIdx.z * seq_len * dim_per_head + blockIdx.y * dim_per_head + threadIdx.x;
    const int k_idx = blockIdx.z * seq_len * dim_per_head + threadIdx.y * dim_per_head + threadIdx.x;
    const int v_idx = blockIdx.z * seq_len * dim_per_head + threadIdx.y * dim_per_head + threadIdx.x;
    const int o_idx = blockIdx.z * seq_len * dim_per_head + blockIdx.y * dim_per_head + threadIdx.x;

    float sum = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        sum += query[q_idx + i * dim_per_head] * key[k_idx + i * dim_per_head];
    }
    sum *= scale;

    float softmax_sum = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        const float exp_val = expf(sum - value[v_idx + i * dim_per_head]);
        softmax_sum += exp_val;
    }

    float attention_score = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        const float exp_val = expf(sum - value[v_idx + i * dim_per_head]);
        attention_score += exp_val / softmax_sum;
        output[o_idx + i * dim_per_head] = attention_score * value[v_idx + i * dim_per_head];
    }
}
