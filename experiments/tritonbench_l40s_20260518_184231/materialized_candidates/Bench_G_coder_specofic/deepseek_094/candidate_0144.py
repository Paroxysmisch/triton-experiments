c
#include <stdint.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

__device__ float apply_bias(float x, float bias, bool softplus) {
    x += bias;
    if (softplus) {
        return log1p(exp(x));
    }
    return x;
}

__device__ float apply_limit(float x, float min, float max) {
    return fmaxf(fminf(x, max), min);
}

__global__ void _chunk_cumsum_fwd_kernel(
    const float* dt, float* dA_cumsum, float* dt_out,
    const float* A, const float* dt_bias, bool dt_softplus,
    float dt_min, float dt_max, int32_t batch, int32_t seqlen, int32_t nheads,
    int32_t chunk_size) {

    int32_t nchunks = (seqlen + chunk_size - 1) / chunk_size;
    int32_t chunk_idx = blockIdx.x * chunk_size + threadIdx.x;
    int32_t head_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t batch_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (chunk_idx >= nchunks || head_idx >= nheads || batch_idx >= batch) {
        return;
    }

    float sum = 0.0f;
    for (int32_t i = 0; i < chunk_size; ++i) {
        int32_t idx = (batch_idx * nheads + head_idx) * seqlen + chunk_idx * chunk_size + i;
        float x = dt[idx];
        float bias = dt_bias ? dt_bias[head_idx] : 0.0f;
        x = apply_bias(x, bias, dt_softplus);
        x = apply_limit(x, dt_min, dt_max);
        dt_out[idx] = x;
        sum += x * A[head_idx];
    }

    dA_cumsum[batch_idx * nheads * nchunks + head_idx * nchunks + chunk_idx] = sum;
}

void _chunk_cumsum_fwd(
    const float* dt, float* dA_cumsum, float* dt_out,
    const float* A, const float* dt_bias, bool dt_softplus,
    float dt_min, float dt_max, int32_t batch, int32_t seqlen, int32_t nheads,
    int32_t chunk_size) {

    dim3 block_dim(16, 16, 1);
    dim3 grid_dim(
        (seqlen + chunk_size - 1) / chunk_size,
        (nheads + 15) / 16,
        (batch + 15) / 16);

    _chunk_cumsum_fwd_kernel<<<grid_dim, block_dim>>>(
        dt, dA_cumsum, dt_out, A, dt_bias, dt_softplus,
        dt_min, dt_max, batch, seqlen, nheads, chunk_size);
}
