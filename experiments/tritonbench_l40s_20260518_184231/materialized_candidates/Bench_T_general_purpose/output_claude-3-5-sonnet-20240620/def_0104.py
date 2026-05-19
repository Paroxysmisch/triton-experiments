import torch
import triton
import triton.language as tl
import math

@triton.jit
def rad2deg_sqrt_kernel(
    input_ptr,      # Pointer to input tensor
    degrees_ptr,    # Pointer to output degrees tensor
    sqrt_ptr,       # Pointer to output sqrt tensor
    n_elements,     # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Number of elements to process per block
):
    # Get program ID
    pid = tl.program_id(axis=0)
    # Calculate start index for this block
    block_start = pid * BLOCK_SIZE
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Convert radians to degrees: angle * (180/pi)
    degrees = x * (180.0 / math.pi)
    # Calculate square root
    sqrt_vals = tl.sqrt(x)
    
    # Store results
    tl.store(degrees_ptr + offsets, degrees, mask=mask)
    tl.store(sqrt_ptr + offsets, sqrt_vals, mask=mask)

def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Input validation
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    # Get input properties
    n_elements = input.numel()
    device = input.device
    
    # Create output tensors
    degrees = torch.empty_like(input)
    sqrt_vals = torch.empty_like(input)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    rad2deg_sqrt_kernel[grid](
        input.contiguous().data_ptr(),
        degrees.data_ptr(),
        sqrt_vals.data_ptr(),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return degrees, sqrt_vals
