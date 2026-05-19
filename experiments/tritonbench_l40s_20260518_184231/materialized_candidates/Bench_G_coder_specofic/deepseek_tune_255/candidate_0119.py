import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr, input_row_stride, output_row_stride, n_cols, **META
):
    BLOCK_SIZE = META["BLOCK_SIZE"]
    row_idx = tl.program_id(axis=0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row_mask = col_offsets < n_cols
    x = tl.load(input_ptrs, mask=row_mask, other=-float("inf"))
    x_minus_max = x - tl.max(x, axis=0)
    num = tl.exp(x_minus_max)
    denom = tl.sum(num, axis=0)
    softmax_out = num / denom
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_out, mask=row_mask)

def triton_softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    output = torch.empty_like(x)
    kernel, grid = softmax_kernel[n_rows,](x, output, x.stride(0), output.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    kernel()
    return output
