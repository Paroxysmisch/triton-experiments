import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    row_start_ptr = input_ptr + row_id * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_pointers = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_pointers, mask=mask, other=-float("inf"))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_id * output_row_stride
    output_pointers = output_row_start_ptr + col_offsets
    tl.store(output_pointers, softmax_output, mask=mask)

def softmax(input: torch.Tensor) -> torch.Tensor:
    n_cols = input.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    if BLOCK_SIZE > 2047:
        num_warps = 8
    if BLOCK_SIZE > 4095:
        num_warps = 16
    output = torch.empty_like(input)
    n_rows = input.numel() // n_cols
    grid = (n_rows,)
    softmax_kernel[grid](
        input,
        output,
        input.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return output
