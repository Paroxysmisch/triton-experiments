import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(
    x_ptr,          # Pointer to input tensor
    output_ptr,     # Pointer to output tensor
    n_elements,     # Number of elements in tensor
    BLOCK_SIZE: tl.constexpr,  # Size of parallel processing block
):
    # Calculate program ID and offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute sigmoid: 1 / (1 + exp(-x))
    output = 1 / (1 + tl.exp(-x))
    
    # Store results
    tl.store(output_ptr + offsets, output, mask=mask)

def sigmoid(input, *, out=None):
    """
    Applies the sigmoid function element-wise.
    
    Args:
        input (Tensor): Input tensor
        out (Tensor, optional): Output tensor. If provided, used to store the result
    
    Returns:
        Tensor: Output tensor with sigmoid activation applied
    """
    # Handle output tensor
    if out is None:
        output = torch.empty_like(input)
    else:
        output = out
        
    # Ensure tensors are on CUDA
    assert input.is_cuda and output.is_cuda, "Input and output tensors must be on CUDA device"
    
    # Get total number of elements
    n_elements = input.numel()
    
    # Define grid
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    sigmoid_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
    
    return output
