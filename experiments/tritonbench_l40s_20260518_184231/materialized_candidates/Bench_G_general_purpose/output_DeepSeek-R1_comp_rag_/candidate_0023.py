import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_f32 = row.to(tl.float32)
    row_max = tl.max(row_f32, axis=0)
    row_minus_max = row_f32 - row_max
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    softmax_output = softmax_output.to(row.dtype)
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def next_power_of_two(n: int) -> int:
    if n == 0:
        return 1
    return 1 << (n - 1).bit_length()

def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2, "Input must be a 2D tensor"
    n_rows, n_cols = x.shape
    block_size = next_power_of_two(n_cols)
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    if block_size >= 4096:
        num_warps = 16
    output = torch.empty_like(x)
    grid = (n_rows,)
    softmax_kernel[grid](
        output,
        x,
        x.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output
