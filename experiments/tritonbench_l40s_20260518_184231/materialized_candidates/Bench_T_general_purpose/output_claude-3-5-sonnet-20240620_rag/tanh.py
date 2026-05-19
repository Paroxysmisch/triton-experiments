import torch
import triton
import triton.language as tl

@triton.jit
def tanh_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(axis=0)
    # Calculate offsets for the current block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask for handling the edge cases
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute tanh using the formula: (e^x - e^-x)/(e^x + e^-x)
    pos_exp = tl.exp(x)
    neg_exp = tl.exp(-x)
    result = (pos_exp - neg_exp) / (pos_exp + neg_exp)
    
    # Store the result
    tl.store(output_ptr + offsets, result, mask=mask)

def tanh(input, *, out=None):
    # Input validation
    assert input.is_cuda, "Input tensor must be on GPU"
    
    # Handle the output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_cuda, "Output tensor must be on GPU"
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
    
    # Calculate total number of elements
    n_elements = input.numel()
    
    # Calculate grid and block size
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 1024))
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch the kernel
    tanh_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
