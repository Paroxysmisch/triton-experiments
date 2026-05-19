import torch
import triton
import triton.language as tl

# Kernel for out-of-place multiplication
@triton.jit
def mul2_kernel(
    x_ptr,  # pointer to input tensor
    output_ptr,  # pointer to output tensor
    n_elements,  # number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # size of parallel blocks
):
    # Compute the pid and the starting index for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle the case where n_elements is not a multiple of BLOCK_SIZE
    mask = offsets < n_elements
    
    # Load data from x
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Perform multiplication
    output = x * 2
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

# Kernel for in-place multiplication
@triton.jit
def mul2_inplace_kernel(
    x_ptr,  # pointer to input/output tensor
    n_elements,  # number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # size of parallel blocks
):
    # Compute the pid and the starting index for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle the case where n_elements is not a multiple of BLOCK_SIZE
    mask = offsets < n_elements
    
    # Load data from x
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Perform multiplication in-place
    x = x * 2
    
    # Store the result back to the same location
    tl.store(x_ptr + offsets, x, mask=mask)

# Python wrapper for out-of-place multiplication
def triton_mul2(x: torch.Tensor) -> torch.Tensor:
    # Get input size and create output tensor
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    # Define block size (can be tuned for performance)
    BLOCK_SIZE = 1024
    
    # Compute grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    mul2_kernel[grid](
        x_ptr=x.data_ptr(),
        output_ptr=output.data_ptr(),
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Python wrapper for in-place multiplication
def triton_mul2_inplace(x: torch.Tensor) -> None:
    # Get input size
    n_elements = x.numel()
    
    # Define block size (can be tuned for performance)
    BLOCK_SIZE = 1024
    
    # Compute grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    mul2_inplace_kernel[grid](
        x_ptr=x.data_ptr(),
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
