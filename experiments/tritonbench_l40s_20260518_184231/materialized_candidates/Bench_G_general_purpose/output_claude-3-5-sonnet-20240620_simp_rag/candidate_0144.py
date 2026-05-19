import torch
import triton
import triton.language as tl

# Kernel for out-of-place multiplication by 2
@triton.jit
def mul2_kernel(
    x_ptr,  # pointer to input tensor
    y_ptr,  # pointer to output tensor
    n_elements,  # number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # number of elements per block
):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block start index
    block_start = pid * BLOCK_SIZE
    
    # Generate offsets for current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load data using mask
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Multiply by 2
    y = x * 2
    
    # Store result using mask
    tl.store(y_ptr + offsets, y, mask=mask)

# Kernel for in-place multiplication by 2
@triton.jit
def mul2_inplace_kernel(
    x_ptr,  # pointer to input/output tensor
    n_elements,  # number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # number of elements per block
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Multiply by 2 in-place
    x = x * 2
    
    # Store result back to same location
    tl.store(x_ptr + offsets, x, mask=mask)

# Wrapper function for out-of-place multiplication
def triton_mul2(x: torch.Tensor) -> torch.Tensor:
    # Input validation
    assert x.is_cuda and x.is_contiguous()
    
    # Create output tensor
    y = torch.empty_like(x)
    
    # Calculate grid size
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    
    # Launch kernel
    mul2_kernel[grid](
        x_ptr=x.data_ptr(),
        y_ptr=y.data_ptr(),
        n_elements=x.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y

# Wrapper function for in-place multiplication
def triton_mul2_inplace(x: torch.Tensor) -> None:
    # Input validation
    assert x.is_cuda and x.is_contiguous()
    
    # Calculate grid size
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    
    # Launch kernel
    mul2_inplace_kernel[grid](
        x_ptr=x.data_ptr(),
        n_elements=x.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )
