c
#include <triton/api.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

__global__ void _quantize_global(const float* x_ptr, const float* absmax_inv_ptr, uint8_t* output_ptr, size_t n_elements) {
    size_t pid = triton::program_id(0);
    size_t block_start = pid * BLOCK_SIZE;
    size_t offsets = blockIdx.x * BLOCK_SIZE;

    if (block_start >= n_elements) return;

    size_t elements_to_process = min(BLOCK_SIZE, n_elements - block_start);
    for (size_t i = threadIdx.x; i < elements_to_process; i++) {
        float value = x_ptr[block_start + offsets + i];
        float scaled = value * (*absmax_inv_ptr);
        output_ptr[block_start + offsets + i] = llrint(scaled);
    }
}

void quantize_global(const float* x, size_t n_elements, float* absmax, uint8_t* output) {
    // Calculate absmax
    *absmax = calculate_absmax(x, n_elements);

    // Calculate absmax_inv
    float absmax_inv = 1.0f / *absmax;

    // Initialize output tensor
    cudaMemset(output, 0, n_elements * sizeof(uint8_t));

    // Dispatch Triton kernel
    dim3 grid_size((n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    _quantize_global<<<grid_size, BLOCK_SIZE>>>(x, &absmax_inv, output, n_elements);
}
