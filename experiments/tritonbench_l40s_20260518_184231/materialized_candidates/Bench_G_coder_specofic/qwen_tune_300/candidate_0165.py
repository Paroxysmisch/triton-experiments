import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, row_stride, n_cols, mask_ptr, BLOCK_SIZE: tl.constexpr):
    # Determine the row index and set up pointers for the row
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # Load the row into SRAM, apply mask if provided, and handle out-of-bounds columns
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    if mask_ptr is not None:
        mask_row = tl.load(mask_ptr + col_offsets)
        row = tl.where(mask_row, row, -float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    
    # Write back the softmax result to DRAM
    output_row_start_ptr = output_ptr + row_idx * row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

def softmax(input: torch.Tensor, mask: torch.Tensor = None, dim=-1) -> torch.Tensor:
    # Ensure the input tensor is valid and reshape it if necessary
    assert input.dim() >= 2, "Input tensor must have at least 2 dimensions"
    assert (mask is None) or mask.shape == input.shape, "Mask shape must match input shape"
    assert dim == -1, "Dimension must be the last axis"
    
    reshaped = input.unsqueeze(0) if input.dim() == 2 else input
    reshaped = reshaped.flatten(0, -2)
    n_rows, n_cols = reshaped.shape
    
    # Define grid and block size for the kernel launch
    BLOCK_SIZE = 1024
    num_warps = 8
    grid = lambda meta: (n_rows,)
    
    # Prepare output tensor and invoke the Triton kernel
    output = torch.empty_like(reshaped)
    softmax_kernel[grid](output, reshaped, reshaped.stride(0), n_cols, mask, BLOCK_SIZE, num_warps=num_warps)
    
    # Reshape the output back to the original input shape
    result = output.view_as(input)
    return result
