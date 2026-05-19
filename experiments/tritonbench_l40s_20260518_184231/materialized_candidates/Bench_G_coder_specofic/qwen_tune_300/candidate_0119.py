import torch
import triton
import triton.language as tl
import math

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for computing the softmax function over a row vector.
    The input and output must be 2D tensors, and the softmax is computed over the rows.
    """
    row_idx = tl.program_id(axis=0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row_mask = col_offsets < n_cols

    row = tl.load(input_ptrs, mask=row_mask, other=-float("inf"))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=row_mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for the Triton softmax kernel.
    This function sets up the kernel arguments and launches it.
    """
    assert x.ndim == 2, "Input tensor must be 2D"
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    BLOCK_SIZE = min(math.ceil(2 ** (math.ceil(math.log2(n_cols)))), 1024)
    grid = lambda meta: (n_rows,)
    softmax_kernel[grid](x, output, x.stride(0), output.stride(0), n_cols, BLOCK_SIZE)

    return output
