import triton
import triton.language as tl
import torch

@triton.jit
def masked_select_kernel(
    inp_ptr,          # pointer to input tensor
    select_mask_ptr,  # pointer to selection mask
    prefix_sum_ptr,   # pointer to prefix sum array
    out_ptr,         # pointer to output tensor
    n_elements,      # total number of elements
    BLOCK_SIZE: tl.constexpr,  # size of parallel blocks
):
    # Calculate global ID for this thread
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input values and selection mask
    x = tl.load(inp_ptr + offsets, mask=mask)
    select = tl.load(select_mask_ptr + offsets, mask=mask)
    prefix = tl.load(prefix_sum_ptr + offsets, mask=mask)
    
    # Convert mask to boolean
    select = select.to(tl.int1)
    
    # Store selected elements to output
    output_idx = prefix - 1
    tl.store(out_ptr + output_idx, x, mask=mask & select)

def masked_select(input_tensor, mask):
    """
    Wrapper function for masked_select_kernel
    Args:
        input_tensor: Input tensor
        mask: Boolean mask tensor
    Returns:
        Selected elements tensor
    """
    # Ensure inputs are on GPU and have compatible shapes
    assert input_tensor.device.type == 'cuda'
    assert mask.device.type == 'cuda'
    
    # Flatten inputs
    input_flat = input_tensor.reshape(-1)
    mask_flat = mask.reshape(-1)
    
    # Compute prefix sum for output positions
    prefix_sum = torch.cumsum(mask_flat.int(), dim=0)
    output_size = int(prefix_sum[-1].item())
    
    # Allocate output tensor
    output = torch.empty(output_size, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Configure kernel parameters
    BLOCK_SIZE = 1024
    grid_size = triton.cdiv(input_flat.numel(), BLOCK_SIZE)
    
    # Launch kernel
    masked_select_kernel[grid_size, BLOCK_SIZE](
        input_flat.data_ptr(),
        mask_flat.data_ptr(),
        prefix_sum.data_ptr(),
        output.data_ptr(),
        input_flat.numel(),
        BLOCK_SIZE,
    )
    
    return output

# Configuration generator for auto-tuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32),
    ],
    key=['n_elements'],
)
def get_configs():
    return 'BLOCK_SIZE'
