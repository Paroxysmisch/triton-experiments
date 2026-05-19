import triton
import triton.language as tl
import torch

# Define block size as a constant
BLOCK_SIZE = 1024

@triton.jit
def mul2_kernel(
    x_ptr,  # Pointer to input tensor
    y_ptr,  # Pointer to output tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Block size (static)
):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load data using mask
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Multiply by 2
    y = x * 2.0
    
    # Store result
    tl.store(y_ptr + offsets, y, mask=mask)

@triton.jit
def mul2_inplace_kernel(
    x_ptr,  # Pointer to input/output tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Block size (static)
):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load data using mask
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Multiply by 2 in-place
    x = x * 2.0
    
    # Store result back to same location
    tl.store(x_ptr + offsets, x, mask=mask)

# Wrapper function for out-of-place multiplication
def triton_mul2(x: torch.Tensor) -> torch.Tensor:
    # Create output tensor
    output = torch.empty_like(x)
    
    # Calculate number of elements
    n_elements = x.numel()
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    mul2_kernel[grid](
        x_ptr=x.data_ptr(),
        y_ptr=output.data_ptr(),
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Wrapper function for in-place multiplication
def triton_mul2_inplace(x: torch.Tensor) -> None:
    # Calculate number of elements
    n_elements = x.numel()
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    mul2_inplace_kernel[grid](
        x_ptr=x.data_ptr(),
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
