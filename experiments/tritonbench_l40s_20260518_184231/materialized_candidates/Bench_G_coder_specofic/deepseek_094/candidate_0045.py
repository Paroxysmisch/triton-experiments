cpp
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <triton/language.h>

namespace cg = cooperative_groups;

__device__ float pow_func_scalar_tensor_kernel_rank_1(float input, float scalar) {
    return powf(input, scalar);
}

__global__ void pow_func_scalar_tensor_wrapper_rank_1(float* input, float* output, float scalar, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = pow_func_scalar_tensor_kernel_rank_1(input[idx], scalar);
    }
}

void pow_func_scalar_tensor_wrapper_rank_1_triton(void* input, void* output, float scalar, int size) {
    dim3 grid((size + 255) / 256);
    dim3 block(256);
    pow_func_scalar_tensor_wrapper_rank_1<<<grid, block>>>((float*)input, (float*)output, scalar, size);
}
