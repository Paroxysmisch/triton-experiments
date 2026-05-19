cpp
#include <cuda_triton.h>

#define BLOCK_M 16
#define BLOCK_N 16

__device__ float* Q_ptr;
__device__ float* K_ptr;
__device__ float* V_ptr;
__device__ float* O_ptr;

__device__ float q_scale;
__device__ float k_scale;

__device__ void _attn_fwd_inner(int m, int n) {
    float acc[BLOCK_N] = {0.0f};
    float l_i[BLOCK_N] = {0.0f};
    float m_i[BLOCK_N] = {-1e30f};

    for (int k = 0; k < n; ++k) {
        float q = Q_ptr[m * n + k];
        float k = K_ptr[m * n + k] * k_scale;
        float v = V_ptr[m * n + k];

        float dot = q * k;
        acc[k] += dot;
        l_i[k] += dot * dot;
        m_i[k] = max(m_i[k], v);
    }

    for (int k = 0; k < n; ++k) {
        l_i[k] = sqrt(l_i[k]);
        acc[k] /= l_i[k];
        acc[k] -= m_i[k];
    }

    O_ptr[m] = acc[0];
}

__device__ void _attn_fwd() {
    int m = blockIdx.x * BLOCK_M;
    int n = blockIdx.y * BLOCK_N;

    for (int i = 0; i < BLOCK_M; ++i) {
        for (int j = 0; j < BLOCK_N; ++j) {
            _attn_fwd_inner(m + i, n + j);
        }
    }
}

__global__ void forward(float* Q, float* K, float* V, float* O, float q_scale, float k_scale) {
    Q_ptr = Q;
    K_ptr = K;
    V_ptr = V;
    O_ptr = O;
    _attn_fwd();
}
