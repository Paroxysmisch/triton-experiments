c
__device__
float2 norm_step(float2 sum, float2 sq_sum, float val) {
    sum.x += val;
    sq_sum.x += val * val;
    return make_float2(sum.x, sq_sum.x);
}

__device__
float2 norm_finalize(float2 sum, float2 sq_sum, int count) {
    float mean = sum.x / count;
    float var = sq_sum.x / count - mean * mean;
    return make_float2(mean, rsqrtf(var + 1e-5f));
}

__global__
void _layer_norm_fwd_1pass_kernel(float* X, float* Y, float* W, float* B, float* RESIDUAL, float* X1, float* W1, float* B1, float* DROPOUT_MASK, int* SEEDS, int M, int N, bool ROWSCALE, bool residual_out) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    float2 sum = make_float2(0.0f, 0.0f);
    float2 sq_sum = make_float2(0.0f, 0.0f);
    for (int i = 0; i < N; ++i) {
        float val = X[m * N + i];
        sum = norm_step(sum, sq_sum, val);
    }
    float2 stats = norm_finalize(sum, sq_sum, N);

    float output = (X[m * N + n] - stats.x) * stats.y;
    output = W[n] * output + B[n];

    if (residual_out && RESIDUAL != nullptr) {
        RESIDUAL[m * N + n] = X[m * N + n] - stats.x;
    }

    if (ROWSCALE) {
        output *= stats.y;
    }

    Y[m * N + n] = output;
}
