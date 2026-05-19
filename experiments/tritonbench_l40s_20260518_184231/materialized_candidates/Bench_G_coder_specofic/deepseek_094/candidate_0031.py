cpp
#include <cuda_runtime.h>
#include <triton/kernel_interface.h>

#define BLOCK_SIZE 1024
#define EXP_COMPENSATOR (2.0f ** (127 - 15))

__device__ void mul_kernel(const float* src, float* dst, const int idx) {
    dst[idx] = src[idx] * EXP_COMPENSATOR;
}

void launch_mul_kernel(const float* src, float* dst, const int size) {
    const int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(grid_size);
    dim3 block(BLOCK_SIZE);

    for (int i = 0; i < grid_size; ++i) {
        mul_kernel<<<grid, block>>>(src, dst, i * BLOCK_SIZE);
    }
}
