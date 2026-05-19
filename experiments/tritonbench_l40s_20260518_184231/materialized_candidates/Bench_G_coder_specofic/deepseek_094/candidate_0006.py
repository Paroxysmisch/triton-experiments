cpp
#include <cuda_runtime.h>
#include <triton/triton.h>

#define BLOCK_M 32
#define BLOCK_N 32
#define BLOCK_DMODEL 32

#define SM_SCALE 0.17

#define N_CTX 12
#define P_SEQ 20

#define OUT_DTYPE float16

using namespace triton;

__device__ OUT_DTYPE attention(const float* Q, const float* K, const float* V, const float* B0, int m, int n, int dmodel, OUT_DTYPE* acc) {
    // scaled dot-product mechanism with relative positional embeddings
    float* softmax_scores = new float[n];
    float* softmax_values = new float[n];
    float sum = 0.0f;

    for (int i = 0; i < n; i++) {
        softmax_scores[i] = dot_product(Q + m * dmodel, K + i * dmodel, dmodel) * SM_SCALE;
        softmax_values[i] = dot_product(Q + m * dmodel, V + i * dmodel, dmodel);
        sum += softmax_scores[i];
    }

    for (int i = 0; i < n; i++) {
        softmax_scores[i] = expf(softmax_scores[i] - sum);
    }

    sum = 0.0f;
    for (int i = 0; i < n; i++) {
        softmax_values[i] *= softmax_scores[i];
        sum += softmax_values[i];
    }

    // accumulate weighted values
    for (int i = 0; i < dmodel; i++) {
        acc[m * dmodel + i] = dot_product(softmax_values, V + i, n);
    }

    // apply bias
    for (int i = 0; i < dmodel; i++) {
        acc[m * dmodel + i] += B0[i];
    }

    delete[] softmax_scores;
    delete[] softmax_values;

    return *reinterpret_cast<OUT_DTYPE*>(acc + m * dmodel);
}

__global__ void _fwd_kernel_aligned(const float* Q, const float* K, const float* V, const float* B0, OUT_DTYPE* acc) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int dmodel = blockDim.z;

    if (m < BLOCK_M && n < BLOCK_N) {
        attention(Q, K, V, B0, m, n, dmodel, acc);
    }
}

void _attention_rel_h_rel_w_kernel_aligned_device(const float* Q, const float* K, const float* V, const float* B0, OUT_DTYPE* acc) {
    dim3 block_dim(BLOCK_M, BLOCK_N, BLOCK_DMODEL);
    dim3 grid_dim(Q_shape[0] / BLOCK_M, Q_shape[1] / BLOCK_N);

    _fwd_kernel_aligned<<<grid_dim, block_dim>>>(Q, K, V, B0, acc);
}
