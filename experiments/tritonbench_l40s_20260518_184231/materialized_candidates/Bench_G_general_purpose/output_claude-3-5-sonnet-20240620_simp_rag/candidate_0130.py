import triton
import triton.language as tl
import torch

@triton.jit
def sine_kernel(
    x_ptr,          # Pointer to input tensor
    output_ptr,     # Pointer to output tensor
    n_elements,     # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Number of elements to process per block
):
    # Compute the pid (program ID) and the number of elements to process
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create an offset array for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for bounds checking
    mask = offsets < n_elements
    
    # Load the input values using the mask
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute sine
    output = tl.sin(x)
    
    # Store the results
    tl.store(output_ptr + offsets, output, mask=mask)

def call_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to call the Triton kernel
    Args:
        x: Input tensor
    Returns:
        Output tensor containing sine values
    """
    # Make sure input is contiguous and on GPU
    x = x.contiguous()
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Define block size (can be tuned for performance)
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    
    # Launch kernel
    sine_kernel[grid](
        x_ptr=x.data_ptr(),
        output_ptr=output.data_ptr(),
        n_elements=x.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
