cpp
#include <triton/ir.h>
#include <triton/triton.h>

void square_kernel(
    float* input, float* output, 
    const int row_stride_in, const int row_stride_out, 
    const int n_cols, const int BLOCK_SIZE
) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    float* row_in = input + row * row_stride_in;
    float* row_out = output + row * row_stride_out;

    float val = 0.0f;
    if (col < n_cols) {
        val = row_in[col];
    }

    float square_output = val * val;

    if (col < n_cols) {
        row_out[col] = square_output;
    }
}

void square(const float* x, float* y, const int n_rows, const int n_cols) {
    int BLOCK_SIZE = 128;
    int num_warps = (n_cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 grid(n_rows);
    dim3 block(BLOCK_SIZE);

    square_kernel<<<grid, block>>>(x, y, n_cols, n_cols, n_cols, BLOCK_SIZE);
}
