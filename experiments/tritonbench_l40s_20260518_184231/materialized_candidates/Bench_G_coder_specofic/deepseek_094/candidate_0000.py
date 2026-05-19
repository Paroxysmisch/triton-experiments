c++
#include <cuda.h>
#include <triton/api.h>

#define BLOCK 64
#define NUM_BLOCK 4
#define CBLOCK 32
#define NUM_CBLOCK 2

__global__ void _fwd_kernel(float* Q, float* K, float* V, float* O, int batch_size, int seq_len, int head_dim) {
    int b = blockIdx.x / head_dim;
    int h = blockIdx.x % head_dim;
    int e = threadIdx.x;
    int s = e / BLOCK;
    int bs = b * seq_len;
    int ks = h * head_dim;

    float Q_part[BLOCK];
    float K_part[BLOCK];
    float V_part[BLOCK];
    float O_part = 0.0f;

    #pragma unroll
    for (int i = 0; i < NUM_BLOCK; i++) {
        if (s * BLOCK + e < seq_len) {
            Q_part[e] = Q[bs + s * BLOCK + e + ks];
            K_part[e] = K[bs + s * BLOCK + e + ks];
            V_part[e] = V[bs + s * BLOCK + e + ks];
        }
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < BLOCK; j++) {
            O_part += Q_part[e] * K_part[j];
        }
        __syncthreads();
    }

    if (s * BLOCK + e < seq_len) {
        O[bs + s * BLOCK + e + ks] = O_part;
    }
}

__global__ void _bwd_intra_kernel(float* Q, float* K, float* V, float* DO, float* DQ, float* DK, float* DV, int batch_size, int seq_len, int head_dim) {
    int b = blockIdx.x / head_dim;
    int h = blockIdx.x % head_dim;
    int e = threadIdx.x;
    int s = e / CBLOCK;
    int bs = b * seq_len;
    int ks = h * head_dim;

    float DO_part = DO[bs + s * CBLOCK + e + ks];
    float Q_part[CBLOCK];
    float K_part[CBLOCK];
    float V_part[CBLOCK];

    #pragma unroll
    for (int i = 0; i < NUM_CBLOCK; i++) {
        if (s * CBLOCK + e < seq_len) {
            Q_part[e] = Q[bs + s * CBLOCK + e + ks];
            K_part[e] = K[bs + s * CBLOCK + e + ks];
            V_part[e] = V[bs + s * CBLOCK + e + ks];
        }
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < CBLOCK; j++) {
            DQ[bs + s * CBLOCK + e + ks] += DO_part * K_part[j];
            DK[bs + s * CBLOCK + e + ks] += Q_part[e] * DO_part;
            DV[bs + s * CBLOCK + e + ks] += DO_part * Q_part[e];
        }
        __syncthreads();
    }
}

__global__ void _bwd_inter_kernel(float* Q, float* K, float* V, float* DO, float* DQ, float* DK, float* DV, int batch_size, int seq_len, int head_dim) {
    int b = blockIdx.x / head_dim;
    int h = blockIdx.x % head_dim;
    int e = threadIdx.x;
    int bs = b * seq_len;
    int ks = h * head_dim;

    float DQ_part = 0.0f;
    float DK_part = 0.0f;
    float DV_part = 0.0f;

    #pragma unroll
    for (int s = 0; s < seq_len; s++) {
        DQ_part += DQ[bs + s * BLOCK + e + ks];
        DK_part += DK[bs + s * BLOCK + e + ks];
        DV_part += DV[bs + s * BLOCK + e + ks];
    }

    if (e < head_dim) {
        DQ[bs + e + ks] = DQ_part;
        DK[bs + e + ks] = DK_part;
        DV[bs + e + ks] = DV_part;
    }
}
