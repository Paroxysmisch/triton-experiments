import triton
import triton.language as tl
import torch

@triton.jit
def _seeded_dropout(x_ptr: ptr, output_ptr: ptr, n_elements: int32, p: float32, seed: int32, BLOCK_SIZE: int32):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    offsets = offsets.clip(0, n_elements - 1)
    x = tl.load(x_ptr + offsets)
    
    # Initialize random number generator
    state = tl.rand.get_state(seed, pid)
    random_numbers = tl.rand.state_to_uniform(state)
    
    # Create mask
    mask = random_numbers > p
    
    # Scale elements
    scaled_elements = x * (1.0 / (1.0 - p))
    
    # Apply mask
    output = tl.where(mask, scaled_elements, 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, output)

def seeded_dropout(x, p, seed, BLOCK_SIZE=1024):
    # Ensure input tensor is contiguous
    x = x.contiguous()
    
    # Create output tensor of the same shape and type as the input tensor
    output = torch.empty_like(x)
    
    # Calculate grid size
    grid_size = (x.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch Triton kernel
    _seeded_dropout[grid_size, BLOCK_SIZE](x.data_ptr(), output.data_ptr(), x.numel(), p, seed, BLOCK_SIZE)
    
    return output
