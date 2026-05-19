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
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    input_row = input_ptr + row_idx * input_row_stride
    output_row = output_ptr + row_idx * output_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load the row, masking out-of-bound elements and replacing with -inf
    row = tl.load(input_row + col_offsets, mask=mask, other=-float('inf'))

    # Compute numerical stable softmax
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Store the result, masking out-of-bound elements
    tl.store(output_row + col_offsets, softmax_output, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    # Compute the next power of two block size up to 1024
    block_size = 1
    while block_size < n_cols and block_size < 1024:
        block_size <<= 1
    block_size = min(block_size, 1024)
    
    # Configure grid and launch kernel
    grid = (n_rows,)
    softmax_kernel[grid](
        x, output,
        x.stride(0), output.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
    )
    return output
