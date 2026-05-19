import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(
    x_ptr,  # pointer to input tensor
    out_ptr,  # pointer to output tensor
    n_elements,  # number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # size of the block for parallel processing
):
    # Calculate the absolute position of the block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create a mask for valid elements in the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values using the mask
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute sigmoid: 1 / (1 + exp(-x))
    output = 1 / (1 + tl.exp(-x))
    
    # Store the result
    tl.store(out_ptr + offsets, output, mask=mask)

def sigmoid(input, *, out=None):
    """
    Applies the sigmoid function element-wise.
    
    Args:
        input (torch.Tensor): Input tensor
        out (torch.Tensor, optional): Output tensor. If provided, the result will be written to this tensor.
                                    Must be the same shape as input.
    
    Returns:
        torch.Tensor: Output tensor with sigmoid activation applied
    """
    # Handle output tensor
    if out is None:
        out = torch.empty_like(input)
    elif out.shape != input.shape:
        raise ValueError(f"Expected out tensor to have shape {input.shape}, but got {out.shape}")
    
    # Get input properties
    n_elements = input.numel()
    
    # Define block size (can be tuned for better performance)
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    sigmoid_kernel[grid](
        input,
        out,
        n_elements,
        BLOCK_SIZE,
    )
    
    return out
