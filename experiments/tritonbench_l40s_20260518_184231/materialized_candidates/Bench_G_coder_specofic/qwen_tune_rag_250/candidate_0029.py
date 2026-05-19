import torch
import triton
import triton.language as tl
from .triton_utils import get_kernel_meta

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
    row_f32 = row.to(tl.float32)
    row_minus_max = row_f32 - tl.max(row_f32, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output.to(row.dtype), mask=col_offsets < n_cols)

def softmax(input):
    n_cols = input.size(-1)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4 if BLOCK_SIZE < 2048 else 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    n_rows = input.numel() // n_cols
    output = torch.empty_like(input)
    kernel_meta = get_kernel_meta(input)
    grid = (n_rows,)
    softmax_kernel[grid](output, input, input.stride(-2), output.stride(-2), n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, **kernel_meta)
    return output
