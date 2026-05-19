import torch
import triton
import triton.language as tl

@triton.jit
def mul_kernel(
    src_ptr,  # Pointer to source tensor
    dst_ptr,  # Pointer to destination tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements to process per block
):
    # Calculate compensator constant (2^(127-15))
    COMPENSATOR = 2.0 ** (127 - 15)
    
    # Calculate the starting position for this program instance
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset array for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements (in case n_elements isn't divisible by BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data from source tensor
    x = tl.load(src_ptr + offsets, mask=mask)
    
    # Multiply by compensator
    output = x * COMPENSATOR
    
    # Store result in destination tensor
    tl.store(dst_ptr + offsets, output, mask=mask)

def launch_mul_kernel(src: torch.Tensor, BLOCK_SIZE: int = 1024):
    """
    Wrapper function to launch the multiplication kernel
    
    Args:
        src: Input tensor
        BLOCK_SIZE: Number of elements to process per block
    
    Returns:
        dst: Output tensor containing results
    """
    # Ensure input is on GPU
    assert src.is_cuda, "Input tensor must be on GPU"
    
    # Create output tensor with same shape and dtype as input
    dst = torch.empty_like(src)
    
    # Calculate grid size
    n_elements = src.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    mul_kernel[grid](
        src_ptr=src.data_ptr(),
        dst_ptr=dst.data_ptr(),
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return dst
