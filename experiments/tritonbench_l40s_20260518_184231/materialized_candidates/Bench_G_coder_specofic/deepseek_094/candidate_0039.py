cpp
__device__
void _fwd_kernel_int8kv(
    int8_t* Q, int8_t* K, int8_t* V, int8_t* Out,
    int H, int BLOCK_DMODEL, int BLOCK_M, int BLOCK_N) {

    int h = blockIdx.y;
    int b = blockIdx.z;
    int m = threadIdx.y;
    int n = threadIdx.z;

    int Q_offset = (b * H + h) * BLOCK_DMODEL * BLOCK_M + m * BLOCK_N + n;
    int K_offset = (b * H + h) * BLOCK_DMODEL * BLOCK_M + m * BLOCK_N + n;
    int V_offset = (b * H + h) * BLOCK_DMODEL * BLOCK_N + m * BLOCK_N + n;
    int Out_offset = (b * H + h) * BLOCK_DMODEL * BLOCK_N + m * BLOCK_N + n;

    int8_t Q_val = Q[Q_offset];
    int8_t K_val = K[K_offset];
    int8_t V_val = V[V_offset];

    // Compute attention scores
    int32_t score = Q_val * K_val;

    // Softmax scaling
    score = score * softmax_scale;

    // Apply causal mask
    if (m < n) {
        score = 0;
    }

    // Apply attention weights to values
    int32_t attention_val = score * V_val;

    // Store attention output
    Out[Out_offset] = attention_val;
}
