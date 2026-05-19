c
__device__
void load_values(float* p_value, float* v_value, float* prob, float* v, int index) {
    p_value[index] = prob[index];
    v_value[index] = v[index];
}

__device__
void accumulate(float* acc, float* p_value, float* v_value, int index) {
    acc[index] += p_value[index] * v_value[index];
}

__global__
void _fwd_kernel_token_att2(float* Prob, float* V, float* Out, int* Req_to_tokens, int batch_size, int seq_len, int num_heads, int head_dim, int* strides, int* indices) {
    int cur_batch = blockIdx.x / num_heads;
    int cur_head = blockIdx.x % num_heads;

    int start_index = cur_head * head_dim;
    int end_index = start_index + head_dim;

    float p_value[BLOCK];
    float v_value[BLOCK];
    float acc[BLOCK];

    for (int i = threadIdx.x; i < BLOCK; i += blockDim.x) {
        int index = indices[i];
        load_values(p_value, v_value, Prob + start_index, V + start_index, index);
        acc[i] = 0.0f;
    }

    __syncthreads();

    for (int i = threadIdx.x; i < BLOCK; i += blockDim.x) {
        int index = indices[i];
        accumulate(acc, p_value, v_value, index);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < BLOCK; i++) {
            int index = indices[i];
            Out[strides[cur_batch] + cur_head * seq_len + Req_to_tokens[index]] = acc[i];
        }
    }
}
