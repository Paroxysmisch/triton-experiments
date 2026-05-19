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
    Softmax kernel, performs softmax across a row of a matrix.
    The matrix is assumed to be stored in column-major order in global memory.
    The function processes one row at a time, using a block of threads to compute
    the softmax across all columns in the row.
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

def softmax(input: torch.Tensor) -> torch.Tensor:
    """
    Compute the softmax of a matrix.

    The matrix is expected to be of shape (n_rows, n_cols) and is processed
    row-wise. The function returns a tensor of the same shape containing the softmax
    probabilities.
    """
    n_rows, n_cols = input.shape

    BLOCK_SIZE = 2 ** (math.ceil(math.log2(n_cols)))
    num_warps = 4
    if BLOCK_SIZE > 2047:
        num_warps = 8
    if BLOCK_SIZE > 4095:
        num_warps = 16

    output = torch.empty_like(input)

    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmax_kernel.warmup(
            input,
            output,
            input.stride(0),
            output.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_programs=n_rows,
            num_warps=num_warps,
        )
        kernel._init_handles()
        kernels[BLOCK_SIZE] = (kernel, n_rows)

    kernel[(n_rows, 1, 1)](
        input,
        output,
        input.stride(0),
        output.stride(0),
        n_cols,
    )

    return output
