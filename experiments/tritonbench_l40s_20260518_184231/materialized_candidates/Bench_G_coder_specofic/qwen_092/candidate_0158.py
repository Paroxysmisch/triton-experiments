cpp
#include <triton/kernel.h>

// Triton kernel function for RMS normalization
__global__ void rmsnorm_triton(const float* x_ptr, const float* rms_w_ptr, float* output_ptr,
                               const int32_t* x_strides, const int32_t* rms_w_strides, const int32_t* output_strides,
                               const int32_t N_SIZE, const float eps, const int32_t BLOCK_N_SIZE) {
    int32_t pid_batch = blockIdx.x;
    int32_t pid_m = blockIdx.y;
    int32_t pid_n = threadIdx.x;

    __shared__ float variance[BLOCK_N_SIZE];

    float sum_squares = 0.0f;
    for (int32_t n = pid_n; n < N_SIZE; n += blockDim.x) {
        float val = x_ptr[pid_batch * x_strides[0] + pid_m * x_strides[1] + n * x_strides[2]];
        sum_squares += val * val;
    }
    __syncthreads();

    int32_t tid = threadIdx.x;
    int32_t warp_id = tid / 32;
    int32_t lane_id = tid % 32;

    variance[warp_id] = sum_squares / N_SIZE;
    __syncthreads();

    if (lane_id == 0) {
        variance[warp_id] = variance[warp_id] + eps;
        variance[warp_id] = 1.0f / sqrt(variance[warp_id]);
    }
    __syncthreads();

    for (int32_t n = pid_n; n < N_SIZE; n += blockDim.x) {
        float val = x_ptr[pid_batch * x_strides[0] + pid_m * x_strides[1] + n * x_strides[2]];
        float normalized_val = val * variance[warp_id] * rms_w_ptr[n * rms_w_strides[0]];
        output_ptr[pid_batch * output_strides[0] + pid_m * output_strides[1] + n * output_strides[2]] = normalized_val;
    }
}

// Wrapper function for launching the RMS normalization kernel
void rmsnorm_triton_wrapper(const float* x_ptr, const float* rms_w_ptr, float* output_ptr,
                             const int32_t* x_strides, const int32_t* rms_w_strides, const int32_t* output_strides,
                             const int32_t batch_size, const int32_t M, const int32_t N_SIZE, const float eps,
                             const int32_t BLOCK_N_SIZE) {
    dim3 grid_size(batch_size, M);
    dim3 block_size(BLOCK_N_SIZE, 1, 1);

    rmsnorm_triton<<<grid_size, block_size>>>(x_ptr, rms_w_ptr, output_ptr, x_strides, rms_w_strides, output_strides,
                                               N_SIZE, eps, BLOCK_N_SIZE);
}
