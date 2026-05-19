cpp
#include <triton/triton.h>
#include <triton/ops/ops.h>
#include <triton/runtime/runtime.h>
#include <triton/runtime/cuda/cuda.h>

#define BLOCK_SIZE 256
#define P2 16

__global__ void _quantize_rowwise(const float* x_ptr, int8_t* output_ptr, int32_t* output_maxs, int n_elements) {
    int row = blockIdx.x;
    int row_size = n_elements / gridDim.x;

    __shared__ float max_val[BLOCK_SIZE];
    max_val[threadIdx.x] = -1.0f;

    for (int col = threadIdx.x; col < row_size; col += blockDim.x) {
        float val = fabsf(x_ptr[row * row_size + col]);
        max_val[threadIdx.x] = fmaxf(max_val[threadIdx.x], val);
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            max_val[threadIdx.x] = fmaxf(max_val[threadIdx.x], max_val[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output_maxs[row] = static_cast<int32_t>(max_val[0]);
        int32_t max_val_int = static_cast<int32_t>(max_val[0]);
        float scale = 127.0f / max_val_int;

        for (int col = 0; col < row_size; ++col) {
            float val = x_ptr[row * row_size + col];
            int8_t quantized_val = static_cast<int8_t>(llrint(val * scale));
            output_ptr[row * row_size + col] = quantized_val;
        }
    }
}

__triton__ void quantize_rowwise(const float* x_ptr, int8_t* output_ptr, int32_t* output_maxs, int n_elements) {
    int num_rows = n_elements / (BLOCK_SIZE * P2);
    dim3 grid_size(num_rows);
    dim3 block_size(BLOCK_SIZE);

    _quantize_rowwise<<<grid_size, block_size>>>(x_ptr, output_ptr, output_maxs, n_elements);
}
