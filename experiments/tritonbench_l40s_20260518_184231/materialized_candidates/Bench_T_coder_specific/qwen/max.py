import torch
from typing import Tuple

@torch.jit.script
def max_wrapper(input: torch.Tensor, dim: int, keepdim: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    n_rows, n_cols = input.shape[:dim], input.shape[dim+1:]
    total_cols = n_cols.prod().item()
    
    # Allocate memory for outputs
    max_output = torch.empty(n_rows + n_cols, dtype=input.dtype, device=input.device)
    idx_output = torch.empty(n_rows + n_cols, dtype=torch.int32, device=input.device)
    
    # Launch the Triton kernel
    grid = (n_rows[0], (total_cols + triton.cdiv(block_size, 256) - 1) // (block_size // 256))
    max_kernel[grid](input.data_ptr(), max_output.data_ptr(), idx_output.data_ptr(),
                      n_rows[0], total_cols, block_size=block_size)
    
    # Reshape the outputs if needed
    if keepdim:
        max_output = max_output.view(*n_rows, 1, *n_cols)
        idx_output = idx_output.view(*n_rows, 1, *n_cols)
    else:
        max_output = max_output.view(*n_rows, *n_cols)
        idx_output = idx_output.view(*n_rows, *n_cols)
    
    return max_output, idx_output
