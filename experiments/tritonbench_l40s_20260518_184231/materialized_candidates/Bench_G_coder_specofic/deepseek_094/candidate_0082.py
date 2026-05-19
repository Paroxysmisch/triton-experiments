c++
#include <cuda_runtime.h>
#include <triton/triton.h>

#define BLOCK_M 16
#define BLOCK_N 16

__global__ void load_reduce_kernel(float* x_ptr, float* y_ptr, int stride_xm, int stride_xn, int stride_y) {
    // Declare the block pointer
    triton::experimental::TritonArray<float, 2> x(x_ptr, {stride_xm, stride_xn});

    // Compute the row-wise maxima
    float max_val = -INFINITY;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < stride_xn; i += blockDim.x * gridDim.x) {
        max_val = tl::max(max_val, x[0][i]);
    }

    // Store the output
    if (threadIdx.x == 0) {
        y_ptr[blockIdx.x * stride_y] = max_val;
    }
}

void load_reduce(float* x_ptr, float* y_ptr, int stride_xm, int stride_xn, int stride_y) {
    // Kernel launch parameters
    dim3 blockSize(BLOCK_N);
    dim3 gridSize((stride_xn + blockSize.x - 1) / blockSize.x);

    // Launch the kernel
    load_reduce_kernel<<<gridSize, blockSize>>>(x_ptr, y_ptr, stride_xm, stride_xn, stride_y);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}
