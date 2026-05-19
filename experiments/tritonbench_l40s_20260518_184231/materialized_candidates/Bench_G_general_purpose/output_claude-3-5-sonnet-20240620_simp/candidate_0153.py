import triton
import triton.language as tl
import torch

@triton.jit
def sin_kernel(
    in_ptr0,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Block size (compile-time constant)
):
    # Calculate the program ID and the number of elements to process
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create an offset array for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for valid elements (to handle edge cases)
    mask = offsets < n_elements
    
    # Load input values using the mask
    x = tl.load(in_ptr0 + offsets, mask=mask)
    
    # Compute sine
    output = tl.sin(x)
    
    # Store the results
    tl.store(out_ptr + offsets, output, mask=mask)

def sin_triton(input_tensor):
    """
    Compute element-wise sine using Triton.
    
    Args:
        input_tensor (torch.Tensor): Input tensor
    Returns:
        torch.Tensor: Output tensor containing sine values
    """
    # Make sure input is on GPU
    assert input_tensor.is_cuda, "Input tensor must be on GPU"
    
    # Get tensor shape and create output
    n_elements = input_tensor.numel()
    output = torch.empty_like(input_tensor)
    
    # Define block size (can be tuned for performance)
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    sin_kernel[grid](
        input_tensor,
        output,
        n_elements,
        BLOCK_SIZE,
    )
    
    return output
