import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = row_idx * n_cols

    # Phase 1: Compute max
    max_val = -float('inf')
    for offset in range(0, n_cols, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        input_ptrs = input_ptr + row_start_ptr + cols
        current = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        current_max = tl.max(current, axis=0)
        max_val = tl.maximum(max_val, current_max)
    
    # Phase 2: Compute sum of exp(row - max)
    sum_val = 0.0
    for offset in range(0, n_cols, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        input_ptrs = input_ptr + row_start_ptr + cols
        current = tl.load(input_ptrs, mask=mask)
        current = current - max_val
        exp_current = tl.exp(current)
        sum_val += tl.sum(exp_current, axis=0)
    
    # Phase 3: Compute softmax and store
    for offset in range(0, n_cols, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        input_ptrs = input_ptr + row_start_ptr + cols
        current = tl.load(input_ptrs, mask=mask)
        current = current - max_val
        exp_current = tl.exp(current)
        softmax_output = exp_current / sum_val
        output_ptrs = output_ptr + row_start_ptr + cols
        tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(input, dim, dtype=None) -> torch.Tensor:
    if dtype is not None:
        input = input.to(dtype)
    dim = dim if dim >= 0 else dim + input.dim()
    
    # Transpose the specified dim to the last dimension to facilitate 2D processing
    input_permuted = input.transpose(dim, -1)
    input_contiguous = input_permuted.contiguous()
    input_2d = input_contiguous.view(-1, input_permuted.shape[-1])
    
    # Allocate output tensor
    output_2d = torch.empty_like(input_2d)
    
    # Kernel parameters
    n_rows, n_cols = input_2d.shape
    BLOCK_SIZE = 1024  # Fixed block size for efficient processing
    grid = (n_rows,)
    
    # Launch kernel
    softmax_kernel[grid](output_2d, input_2d, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    # Reshape back to original dimensions
    output_permuted = output_2d.view(input_contiguous.shape)
    output = output_permuted.transpose(dim, -1).contiguous()
    
    return output
