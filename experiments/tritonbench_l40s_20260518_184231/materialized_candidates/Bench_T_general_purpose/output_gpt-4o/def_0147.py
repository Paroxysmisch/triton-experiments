import torch
import triton
import triton.language as tl

@triton.jit
def pairwise_distance_kernel(
    x1_ptr, x2_ptr, out_ptr, 
    n_elements, p_norm, eps_norm, eps_distance,
    BLOCK_SIZE: tl.constexpr
):
    # Compute the row and column indices for the block
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Create a pointer to the start of the block
    x1_offset = row_idx * n_elements
    x2_offset = col_idx * n_elements
    
    # Load the blocks of x1 and x2
    x1 = tl.load(x1_ptr + x1_offset + tl.arange(0, BLOCK_SIZE))
    x2 = tl.load(x2_ptr + x2_offset + tl.arange(0, BLOCK_SIZE))
    
    # Normalize x1 and x2
    norm_x1 = tl.norm(x1, p=p_norm) + eps_norm
    norm_x2 = tl.norm(x2, p=p_norm) + eps_norm
    x1 = x1 / norm_x1
    x2 = x2 / norm_x2
    
    # Compute the pairwise distance
    diff = x1 - x2
    distance = tl.norm(diff, p=p_norm) + eps_distance
    
    # Store the result
    tl.store(out_ptr + row_idx * n_elements + col_idx, distance)

def fused_pairwise_distance_normalize(
    x1: torch.Tensor, x2: torch.Tensor, 
    p_norm: float = 2.0, eps_norm: float = 1e-12, 
    eps_distance: float = 1e-6, keepdim: bool = False
) -> torch.Tensor:
    # Ensure x1 and x2 are on the same device and have the same dtype
    assert x1.device == x2.device, "x1 and x2 must be on the same device"
    assert x1.dtype == x2.dtype, "x1 and x2 must have the same dtype"
    
    # Get the number of elements along the specified dimension
    n_elements = x1.shape[-1]
    
    # Prepare the output tensor
    output_shape = (x1.shape[0], x2.shape[0]) if not keepdim else (x1.shape[0], x2.shape[0], 1)
    out = torch.empty(output_shape, device=x1.device, dtype=x1.dtype)
    
    # Launch the Triton kernel
    grid = (x1.shape[0], x2.shape[0])
    pairwise_distance_kernel[grid](
        x1_ptr=x1, x2_ptr=x2, out_ptr=out,
        n_elements=n_elements, p_norm=p_norm, 
        eps_norm=eps_norm, eps_distance=eps_distance,
        BLOCK_SIZE=n_elements
    )
    
    return out
