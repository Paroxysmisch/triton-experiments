import triton
import triton.language as tl
import torch
import math

@triton.jit
def sub_kernel_tensor(
    input_ptr, other_ptr, output_ptr,
    n_elements, alpha: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Kernel for tensor-tensor subtraction with alpha scaling
    out = input - alpha * other
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.load(other_ptr + offsets, mask=mask)
    
    # Compute subtraction with alpha scaling
    output = x - alpha * y
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def sub_kernel_scalar(
    input_ptr, other: tl.constexpr, output_ptr,
    n_elements, alpha: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Kernel for tensor-scalar subtraction with alpha scaling
    out = input - alpha * other
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute subtraction with alpha scaling
    output = x - alpha * other
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def sub(input, other, *, alpha=1, out=None):
    """
    Subtracts other, scaled by alpha, from input.
    
    Args:
        input (Tensor): the input tensor
        other (Tensor or Number): the tensor or number to subtract from input
        alpha (Number, optional): the multiplier for other. Defaults to 1
        out (Tensor, optional): the output tensor. Defaults to None
    
    Returns:
        Tensor: The result of input - alpha * other
    """
    # Handle output tensor
    if out is None:
        out = torch.empty_like(input)
    
    # Get total number of elements
    n_elements = input.numel()
    
    # Calculate block size and grid size
    block_size = triton.next_power_of_2(min(n_elements, 1024))
    grid_size = triton.cdiv(n_elements, block_size)
    
    # Launch appropriate kernel based on other type
    if isinstance(other, torch.Tensor):
        # Handle broadcasting if needed
        if input.shape != other.shape:
            other = other.expand_as(input)
        sub_kernel_tensor[(grid_size,)](
            input_ptr=input, 
            other_ptr=other,
            output_ptr=out,
            n_elements=n_elements,
            alpha=alpha,
            BLOCK_SIZE=block_size
        )
    else:
        # Scalar case
        sub_kernel_scalar[(grid_size,)](
            input_ptr=input,
            other=other,
            output_ptr=out,
            n_elements=n_elements,
            alpha=alpha,
            BLOCK_SIZE=block_size
        )
    
    return out
