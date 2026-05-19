c++
#include <cuda.h>
#include <cuda_runtime.h>
#include <triton/api.h>

#define BLOCK_SIZE 1024

__device__ void masked_add_kernel(float* grad_ptr, const float* p_ptr, const int* p_mask_ptr, float alpha, int n_elements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n_elements) {
        bool mask = (bool)p_mask_ptr[idx];
        if (mask) {
            grad_ptr[idx] += alpha * p_ptr[idx];
        }
    }
}

void masked_add(const float* grad_ptr, const float* p_ptr, const int* p_mask_ptr, float alpha, int n_elements) {
    assert(grad_ptr && p_ptr && p_mask_ptr);
    assert(cudaPointerGetAttributes(&grad_ptr->memoryType, grad_ptr) == cudaSuccess);
    assert(cudaPointerGetAttributes(&p_ptr->memoryType, p_ptr) == cudaSuccess);
    assert(cudaPointerGetAttributes(&p_mask_ptr->memoryType, p_mask_ptr) == cudaSuccess);

    dim3 gridSize((n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
    masked_add_kernel<<<gridSize, BLOCK_SIZE>>>(grad_ptr, p_ptr, p_mask_ptr, alpha, n_elements);
}
