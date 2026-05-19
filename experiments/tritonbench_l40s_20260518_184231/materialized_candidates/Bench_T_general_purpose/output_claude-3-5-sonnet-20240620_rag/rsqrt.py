import torch
import triton
import triton.language as tl

@triton.jit
def rsqrt_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr, # Pointer to output tensor
    n_elements, # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Number of elements to process per block
):
    # Get the program ID
    pid = tl.program_id(axis=0)
    # Calculate the starting offset for this program instance
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask for valid memory accesses
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute reciprocal square root
    # For negative values, this will automatically result in nan
    result = 1.0 / tl.sqrt(x)
    
    # Store the result
    tl.store(output_ptr + offsets, result, mask=mask)

def rsqrt(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Input validation
    assert input.is_cuda, "Input tensor must be on GPU"
    
    # If output tensor is not provided, create a new one
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_cuda, "Output tensor must be on GPU"
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
    
    # Calculate number of elements
    n_elements = input.numel()
    
    # Calculate grid and block sizes
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 1024))
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    rsqrt_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
