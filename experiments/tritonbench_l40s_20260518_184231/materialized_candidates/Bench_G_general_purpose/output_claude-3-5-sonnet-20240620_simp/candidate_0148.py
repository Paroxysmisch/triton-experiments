import torch
import triton
import triton.language as tl

@triton.jit
def mul2_kernel(
    x_ptr,  # Pointer to input tensor
    y_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements per block
):
    # Calculate the absolute position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle the case where array size is not divisible by BLOCK_SIZE
    mask = offsets < n_elements
    
    # Load data using the mask
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Perform the computation
    output = x * 2.0
    
    # Store the result
    tl.store(y_ptr + offsets, output, mask=mask)

@triton.jit
def mul2_inplace_kernel(
    x_ptr,  # Pointer to input/output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements per block
):
    # Calculate the absolute position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle the case where array size is not divisible by BLOCK_SIZE
    mask = offsets < n_elements
    
    # Load data using the mask
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Perform the computation
    output = x * 2.0
    
    # Store the result back to the same location
    tl.store(x_ptr + offsets, output, mask=mask)

def triton_mul2(x: torch.Tensor) -> torch.Tensor:
    """
    Double each element in the input tensor and return a new tensor.
    
    Args:
        x: Input tensor
    Returns:
        New tensor with doubled values
    """
    output = torch.empty_like(x)
    assert x.is_contiguous(), "Input tensor must be contiguous"
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    mul2_kernel[grid](
        x_ptr=x,
        y_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def triton_mul2_inplace(x: torch.Tensor) -> None:
    """
    Double each element in the input tensor in-place.
    
    Args:
        x: Input tensor to be modified in-place
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    mul2_inplace_kernel[grid](
        x_ptr=x,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
