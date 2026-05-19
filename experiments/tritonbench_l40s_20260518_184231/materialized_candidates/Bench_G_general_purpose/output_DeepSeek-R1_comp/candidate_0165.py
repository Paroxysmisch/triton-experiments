import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    row_stride,
    n_cols,
    mask_ptr,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start + col_offsets
    mask = col_offsets < n_cols

    # Load the row data into SRAM, masking out-of-bounds columns
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    # Compute maximum value for numerical stability
    row_max = tl.max(row, axis=0)
    row_minus_max = row - row_max

    # Apply mask if provided
    if mask_ptr != 0:
        mask_ptrs = mask_ptr + row_idx * row_stride + col_offsets
        mask_values = tl.load(mask_ptrs, mask=mask, other=0)
        row_minus_max += mask_values

    # Compute exponentials and their sum
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Write the result back to output
    output_ptrs = output_ptr + row_idx * row_stride + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(input: torch.Tensor, mask: torch.Tensor = None, dim=-1) -> torch.Tensor:
    # Validate input dimensions
    assert dim == -1 or dim == input.dim() - 1, "Softmax only supported along the last dimension"
    assert input.dim() >= 1, "Input must have at least one dimension"

    if mask is not None:
        assert mask.shape == input.shape, "Mask must have the same shape as input"

    # Flatten the input to 2D if necessary
    if input.dim() > 2:
        input_2d = input.view(-1, input.size(-1))
    else:
        input_2d = input

    n_rows, n_cols = input_2d.shape

    # Determine the block size as the next power of two up to 4096
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE < 32:
        BLOCK_SIZE = 32
    elif BLOCK_SIZE > 4096:
        BLOCK_SIZE = 4096

    # Determine the number of warps for optimal performance
    num_warps = 4 if BLOCK_SIZE >= 4096 else (2 if BLOCK_SIZE >= 2048 else 1)

    # Prepare the mask pointer
    mask_ptr = mask.data_ptr() if mask is not None else 0

    # Allocate output tensor
    output = torch.empty_like(input_2d)

    # Launch the kernel with the computed grid and block parameters
    grid = (n_rows,)
    softmax_kernel[grid](
        output,
        input_2d,
        input_2d.stride(0),
        n_cols,
        mask_ptr,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    # Reshape the output to match the original input shape
    return output.view(input.shape)
