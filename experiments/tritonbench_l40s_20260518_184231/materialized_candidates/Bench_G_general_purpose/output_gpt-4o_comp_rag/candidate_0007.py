import triton
import triton.language as tl

@triton.jit
def masked_select_kernel(inp_ptr, select_mask_ptr, prefix_sum_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate global thread index
    block_id = tl.program_id(0)
    start_idx = block_id * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    
    # Check bounds
    mask = offsets < n_elements
    
    # Load input and mask values
    inp = tl.load(inp_ptr + offsets, mask=mask)
    select_mask = tl.load(select_mask_ptr + offsets, mask=mask)
    
    # Convert mask to boolean
    select_mask_bool = select_mask.to(tl.int1)
    
    # Load prefix sum values
    prefix_sum = tl.load(prefix_sum_ptr + offsets, mask=mask)
    
    # Calculate output positions
    out_positions = prefix_sum - 1
    
    # Store selected values to output buffer
    tl.store(out_ptr + out_positions, inp, mask=select_mask_bool & mask)

def masked_select(inp, select_mask, prefix_sum, BLOCK_SIZE=1024):
    # Ensure input tensors are broadcastable
    assert broadcastable(inp.shape, select_mask.shape), "Shapes are not broadcastable"
    
    # Flatten the mask and compute prefix sum
    select_mask_flat = select_mask.flatten()
    prefix_sum = prefix_sum.flatten()
    
    # Number of elements
    n_elements = inp.numel()
    
    # Allocate output buffer
    out = torch.empty(n_elements, dtype=inp.dtype, device=inp.device)
    
    # Calculate grid size
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    masked_select_kernel[grid_size](inp, select_mask_flat, prefix_sum, out, n_elements, BLOCK_SIZE)
    
    return out

def broadcastable(shape1, shape2):
    """Check if two shapes are broadcastable."""
    for s1, s2 in zip(reversed(shape1), reversed(shape2)):
        if s1 != s2 and s1 != 1 and s2 != 1:
            return False
    return True

# Example usage:
import torch

# Input tensors
inp = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device='cuda')
select_mask = torch.tensor([0, 1, 0, 1, 1], dtype=torch.int32, device='cuda')
prefix_sum = torch.tensor([0, 1, 1, 2, 3], dtype=torch.int32, device='cuda')

# Perform masked select
out = masked_select(inp, select_mask, prefix_sum)
print(out)
