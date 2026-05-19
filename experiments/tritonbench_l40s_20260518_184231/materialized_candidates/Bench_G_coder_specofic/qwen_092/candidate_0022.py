cpp
#include <triton/triton.h>

// Constants
const int BLOCK_M = 128;
const int BLOCK_N = 128;
const int BLOCK_DMODEL = 128;

// Forward kernel
__global__ void _fwd_kernel(
    float* Q, float* K, float* V, float* Out, float* sm_scale, float* mask, float* attention_weights,
    int batch_size, int seq_len, int num_heads, int d_model) {
    __shared__ float q_tile[BLOCK_M][BLOCK_DMODEL];
    __shared__ float k_tile[BLOCK_DMODEL][BLOCK_N];
    __shared__ float v_tile[BLOCK_N][BLOCK_DMODEL];

    int b = blockIdx.x / (seq_len / BLOCK_M);
    int m = blockIdx.x % (seq_len / BLOCK_M);
    int n = blockIdx.y;
    int h = blockIdx.z;

    int q_idx = b * seq_len * num_heads * d_model + m * num_heads * d_model + h * d_model;
    int k_idx = b * seq_len * num_heads * d_model + n * num_heads * d_model + h * d_model;
    int v_idx = b * seq_len * num_heads * d_model + n * num_heads * d_model + h * d_model;
    int out_idx = b * seq_len * num_heads * d_model + m * num_heads * d_model + h * d_model;

    float dot_product = 0.0f;

    // Load Q, K, V tiles
    q_tile[m][lane] = Q[q_idx + lane];
    k_tile[lane][n] = K[k_idx + lane];
    v_tile[n][lane] = V[v_idx + lane];

    __syncthreads();

    // Compute dot product
    for (int d = 0; d < BLOCK_DMODEL; ++d) {
        dot_product += q_tile[m][d] * k_tile[d][n];
    }

    __syncthreads();

    // Scale and apply mask
    dot_product *= sm_scale[0];
    if (mask != nullptr) {
        dot_product *= mask[m * seq_len + n];
    }

    // Compute softmax
    float max_val = -FLT_MAX;
    for (int i = 0; i < seq_len; ++i) {
        float val = dot_product - (i == n ? 0.0f : -FLT_MAX);
        max_val = max(max_val, val);
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        sum_exp += exp(dot_product - max_val - (i == n ? 0.0f : -FLT_MAX));
    }

    float softmax = exp(dot_product - max_val) / sum_exp;

    // Store attention weights
    attention_weights[out_idx] = softmax;

    // Compute weighted sum of V
    float weighted_sum = 0.0f;
    for (int d = 0; d < BLOCK_DMODEL; ++d) {
        weighted_sum += v_tile[n][d] * softmax;
    }

    // Store output
    Out[out_idx] = weighted_sum;
}

// Backward preprocess
__global__ void _bwd_preprocess(
    float* DO, float* L, float* delta, int batch_size, int seq_len, int num_heads, int d_model) {
    int b = blockIdx.x / (seq_len / BLOCK_M);
    int m = blockIdx.x % (seq_len / BLOCK_M);
    int n = blockIdx.y;
    int h = blockIdx.z;

    int do_idx = b * seq_len * num_heads * d_model + m * num_heads * d_model + h * d_model;
    int l_idx = b * seq_len * num_heads * d_model + n * num_heads * d_model + h * d_model;
    int delta_idx = b * seq_len * num_heads * d_model + m * num_heads * d_model + h * d_model;

    float do_val = DO[do_idx];
    float l_val = L[l_idx];

    // Scale DO with normalization constant L
    delta[delta_idx] = do_val * l_val;
}

// Backward kernel
__global__ void _bwd_kernel(
    float* Q, float* K, float* V, float* DO, float* L, float* Q_grad, float* K_grad, float* V_grad,
    float* sm_scale, float* attention_weights, int batch_size, int seq_len, int num_heads, int d_model) {
    __shared__ float q_tile[BLOCK_M][BLOCK_DMODEL];
    __shared__ float k_tile[BLOCK_DMODEL][BLOCK_N];
    __shared__ float v_tile[BLOCK_N][BLOCK_DMODEL];
    __shared__ float attention_weights_tile[BLOCK_M][BLOCK_N];

    int b = blockIdx.x / (seq_len / BLOCK_M);
    int m = blockIdx.x % (seq_len / BLOCK_M);
    int n = blockIdx.y;
    int h = blockIdx.z;

    int q_idx = b * seq_len * num_heads * d_model + m * num_heads * d_model + h * d_model;
    int k_idx = b * seq_len * num_heads * d_model + n * num_heads * d_model + h * d_model;
    int v_idx = b * seq_len * num_heads * d_model + n * num_heads * d_model + h * d_model;
    int do_idx = b * seq_len * num_heads * d_model + m * num_heads * d_model + h * d_model;
    int q_grad_idx = b * seq_len * num_heads * d_model + m * num_heads * d_model + h * d_model;
    int k_grad_idx = b * seq_len * num_heads * d_model + n * num_heads * d_model + h * d_model;
    int v_grad_idx = b * seq_len * num_heads * d_model + n * num_heads * d_model + h * d_model;

    float do_val = DO[do_idx];
    float attention_weight = attention_weights[do_idx];

    // Load Q, K, V tiles
    q_tile[m][lane] = Q[q_idx + lane];
    k_tile[lane][n] = K[k_idx + lane];
    v_tile[n][lane] = V[v_idx + lane];
    attention_weights_tile[m][n] = attention_weight;

    __syncthreads();

    // Compute gradients
    float q_grad = 0.0f;
    float k_grad = 0.0f;
    float v_grad = 0.0f;

    for (int d = 0; d < BLOCK_DMODEL; ++d) {
        q_grad += do_val * k_tile[d][n];
        k_grad += do_val * q_tile[m][d];
        v_grad += do_val * attention_weight * q_tile[m][d];
    }

    // Accumulate gradients
    atomicAdd(&Q_grad[q_grad_idx + lane], q_grad);
    atomicAdd(&K_grad[k_grad_idx + lane], k_grad);
    atomicAdd(&V_grad[v_grad_idx + lane], v_grad);

    __syncthreads();
}

// Attention wrapper class
class _attention {
public:
    _attention(int batch_size, int seq_len, int num_heads, int d_model, float sm_scale, float* mask)
        : batch_size(batch_size), seq_len(seq_len), num_heads(num_heads), d_model(d_model),
          sm_scale(sm_scale), mask(mask) {}

    void forward(float* Q, float* K, float* V, float* Out) {
        int grid_x = (batch_size * seq_len / BLOCK_M) * num_heads;
        int grid_y = seq_len / BLOCK_N;
        int grid_z = num_heads;

        _fwd_kernel<<<grid_x, dim3(BLOCK_M, BLOCK_N, 1)>>>(
            Q, K, V, Out, sm_scale, mask, attention_weights.data(), batch_size, seq_len, num_heads, d_model);
    }

    void backward(float* Q, float* K, float* V, float* DO, float* L, float* Q_grad, float* K_grad, float* V_grad) {
        int grid_x = (batch_size * seq_len / BLOCK_M) * num_heads;
        int grid_y = seq_len / BLOCK_N;
        int grid_z = num_heads;

        _bwd_preprocess<<<grid_x, dim3(BLOCK_M, BLOCK_N, 1)>>>(
            DO, L, delta.data(), batch_size, seq_len, num_heads, d_model);

        _bwd_kernel<<<grid_x, dim3(BLOCK_M, BLOCK_N, 1)>>>(
            Q, K, V, DO, L, Q_grad, K_grad, V_grad, sm_scale, attention_weights.data(), batch_size, seq_len, num_heads, d_model);
    }

private:
    int batch_size, seq_len, num_heads, d_model;
    float sm_scale;
    float* mask;
    std::vector<float> attention_weights;
    std::vector<float> delta;
};
