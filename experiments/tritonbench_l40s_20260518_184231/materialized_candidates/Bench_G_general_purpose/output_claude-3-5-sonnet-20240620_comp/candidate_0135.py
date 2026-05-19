import triton
import triton.language as tl
import torch

@triton.jit
def kernel_function(
    x_ptr,          # Pointer to input tensor
    output_ptr,     # Pointer to output tensor
    n_elements,     # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Static block size for processing
):
    # Calculate starting point for this program instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Create offset tensor for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements

    # Load input data using the mask
    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute sine of input values
    output = tl.math.sin(x)

    # Store results back to memory
    tl.store(output_ptr + offsets, output, mask=mask)

def call_kernel(x: torch.Tensor) -> torch.Tensor:
    # Ensure input is contiguous and on GPU
    x = x.contiguous()
    
    # Get total number of elements
    n_elements = x.numel()
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Define block size
    BLOCK_SIZE = 1024

    # Calculate grid size based on block size
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel
    kernel_function[grid](
        x_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output
