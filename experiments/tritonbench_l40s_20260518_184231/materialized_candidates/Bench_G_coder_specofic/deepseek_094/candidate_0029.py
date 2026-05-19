import triton
import triton.language as tl

@triton.jit
def softmax_kernel(input_ptr, output_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    start_ptr = input_ptr + row * input_row_stride
    grid_stride = BLOCK_SIZE * input_row_stride
    offsets = row * output_row_stride + tl.arange(0, n_cols)
    x = tl.load(start_ptr + offsets, mask=(tl.arange(0, n_cols) < n_cols), other=-1e10)
    x_max = tl.max(x, axis=0)
    x = x - x_max
    exp_x = tl.exp(x)
    sum_exp_x = tl.sum(exp_x, axis=0)
    y = exp_x / sum_exp_x
    tl.store(output_ptr + offsets, y)

def softmax(input_tensor, output_tensor):
    n_rows, n_cols = input_tensor.shape
    BLOCK_SIZE = 1024
    num_warps = -(-n_cols // BLOCK_SIZE)  # Ceiling division
    input_ptr = triton.pointers.register_ptr(input_tensor)
    output_ptr = triton.pointers.register_ptr(output_tensor)
    input_row_stride = input_tensor.strides[0]
    output_row_stride = output_tensor.strides[0]
    softmax_kernel[(num_warps,)](input_ptr, output_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE)
    return output_tensor
