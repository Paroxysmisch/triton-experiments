cpp
__device__
float rms_normalization(float* x, int len) {
    float sum_squares = 0.0f;
    for (int i = 0; i < len; i++) {
        sum_squares += x[i] * x[i];
    }
    float mean_squares = sum_squares / len;
    return rsqrtf(mean_squares);
}

__global__
void rms_matmul_rbe(float* x, float* w, float* rms_w, float* out, int batch_size, int seq_len, int num_heads, int head_size) {
    int batch_id = blockIdx.x / (seq_len * num_heads);
    int seq_id = (blockIdx.x % (seq_len * num_heads)) / num_heads;
    int head_id = blockIdx.x % num_heads;

    int x_offset = batch_id * seq_len * head_size + seq_id * head_size;
    int w_offset = head_id * head_size;
    int out_offset = batch_id * seq_len * num_heads + seq_id * num_heads + head_id;

    float rms_scale = rms_w[head_id];
    float* x_row = x + x_offset;
    float* w_col = w + w_offset;
    float* out_row = out + out_offset;

    float rms_norm = rms_normalization(x_row, head_size);
    for (int i = 0; i < head_size; i++) {
        x_row[i] *= rms_norm * rms_scale;
    }

    for (int i = 0; i < head_size; i++) {
        out_row[i] = 0.0f;
        for (int j = 0; j < head_size; j++) {
            out_row[i] += x_row[j] * w_col[j];
        }
    }
}

void rms_matmul_rbe_wrapper(float* x, float* w, float* rms_w, float* out, int batch_size, int seq_len, int num_heads, int head_size) {
    dim3 grid_dim(batch_size * seq_len * num_heads);
    dim3 block_dim(head_size);
    rms_matmul_rbe<<<grid_dim, block_dim>>>(x, w, rms_w, out, batch_size, seq_len, num_heads, head_size);
}
