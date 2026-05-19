import math
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr, 
    output_ptr, 
    input_row_stride, 
    output_row_stride, 
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    row_input_ptr = input_ptr + row_id * input_row_stride
    row_output_ptr = output_ptr + row_id * output_row_stride

    # Create an index for each element in the row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # Load row data
    row_vals = tl.load(row_input_ptr + cols, mask=mask, other=-float('inf'))

    # Numerical stabilization
    row_max = tl.max(row_vals, axis=0)
    row_vals = row_vals - row_max

    # Exponentiation
    exps = tl.exp(row_vals)
    denom = tl.sum(exps, axis=0)

    # Final softmax
    softmax_vals = exps / denom

    # Store results
    tl.store(row_output_ptr + cols, softmax_vals, mask=mask)


def softmax(input_tensor: torch.Tensor) -> torch.Tensor:
    # Ensure input is contiguous
    input_contig = input_tensor.contiguous()
    n_rows, n_cols = input_contig.shape

    # Allocate output
    output = torch.empty_like(input_contig)

    # Helper to find next power of two
    def next_power_of_two(x):
        return 1 << (x-1).bit_length()

    block_size = next_power_of_two(n_cols)
    num_warps = 4 if block_size >= 128 else 1

    grid = (n_rows,)
    softmax_kernel[grid](
        input_contig,
        output,
        input_contig.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps
    )
    return output
