import triton
import triton.language as tl
import torch

@triton.jit
def airy_ai_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr, # Pointer to output tensor
    n_elements, # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the absolute position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute Airy Ai function using series expansion
    # For x < 0, we use the oscillatory series
    # For x >= 0, we use the exponential decay series
    
    result = tl.where(
        x >= 0,
        _airy_ai_positive(x),
        _airy_ai_negative(x)
    )
    
    # Store results
    tl.store(output_ptr + offsets, result, mask=mask)

@triton.jit
def _airy_ai_positive(x):
    # Constants for the series expansion
    c1 = 0.355028053887817
    c2 = 0.258819403792807
    z = (2.0/3.0) * tl.pow(x, 1.5)
    return (c1 / tl.pow(x, 0.25)) * tl.exp(-z)

@triton.jit
def _airy_ai_negative(x):
    # Constants for the series expansion
    pi = 3.141592653589793
    c3 = 0.355028053887817
    z = (2.0/3.0) * tl.pow(-x, 1.5)
    return (c3 / tl.pow(tl.abs(x), 0.25)) * tl.cos(z - pi/4)

def airy_ai(input, *, out=None):
    """
    Computes the Airy function Ai for each element of the input tensor.
    
    Args:
        input (Tensor): the input tensor
        out (Tensor, optional): the output tensor
    
    Returns:
        Tensor: A tensor containing the Airy Ai function values
    """
    if out is None:
        out = torch.empty_like(input)
    elif out.shape != input.shape:
        raise ValueError("out tensor must have the same shape as input tensor")
    
    # Ensure input is contiguous
    if not input.is_contiguous():
        input = input.contiguous()
    if not out.is_contiguous():
        out = out.contiguous()
    
    # Launch kernel
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    airy_ai_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=1024,
    )
    
    return out
