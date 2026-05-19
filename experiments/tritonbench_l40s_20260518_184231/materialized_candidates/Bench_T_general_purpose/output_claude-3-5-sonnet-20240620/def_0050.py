import torch
import triton
import triton.language as tl

@triton.jit
def sqrt_exp_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr, # Pointer to output tensor
    n_elements, # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process
):
    # Calculate the program ID and define the block of elements to process
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for bounds checking
    mask = offsets < n_elements
    
    # Load input values using the mask
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute sqrt(x) and then exp(sqrt(x))
    x_sqrt = tl.sqrt(x)
    result = tl.exp(x_sqrt)
    
    # Store the result
    tl.store(output_ptr + offsets, result, mask=mask)

def sqrt_exp(input, out=None):
    """
    Computes e^(sqrt(x)) for each element x in the input tensor.
    
    Args:
        input (Tensor): The input tensor
        out (Tensor, optional): The output tensor. If None, a new tensor is created
    
    Returns:
        Tensor: A tensor containing the result
    """
    # Input validation
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch tensor")
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    elif out.shape != input.shape:
        raise ValueError("Output tensor must have the same shape as input tensor")
    
    # Ensure input and output are contiguous
    input = input.contiguous()
    out = out.contiguous()
    
    # Define grid and block sizes
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(input.numel(), BLOCK_SIZE),)
    
    # Launch kernel
    sqrt_exp_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=input.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
