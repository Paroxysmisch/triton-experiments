import triton
import triton.language as tl
import torch
import math

@triton.jit
def gelu_std_kernel(
    input_ptr, output_ptr,
    stride_in, stride_out,
    n_elements, block_size,
    BLOCK_SIZE: tl.constexpr,
    approximate: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets * stride_in, mask=mask)
    
    # GELU activation
    if approximate == 'tanh':
        # Tanh approximation
        cdf = 0.5 * (1.0 + tl.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)
        ))
        gelu_out = x * cdf
    else:
        # Exact GELU using error function
        cdf = 0.5 * (1.0 + tl.erf(x / math.sqrt(2.0)))
        gelu_out = x * cdf
    
    # Store result
    tl.store(output_ptr + offsets * stride_out, gelu_out, mask=mask)

def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None):
    """
    Applies GELU activation and computes standard deviation along specified dimensions.
    
    Args:
        input (Tensor): Input tensor
        dim (int or tuple of ints, optional): Dimension(s) to reduce
        keepdim (bool): Whether to keep reduced dimensions
        correction (int): Bessel's correction factor
        approximate (str): GELU approximation method ('none' or 'tanh')
        out (Tensor, optional): Output tensor
    
    Returns:
        Tensor: Standard deviation of GELU-activated values
    """
    # Input validation
    if approximate not in ['none', 'tanh']:
        raise ValueError("approximate must be 'none' or 'tanh'")
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    
    # Configure grid and block sizes
    BLOCK_SIZE = 1024
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel for GELU activation
    gelu_std_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        stride_in=1,
        stride_out=1,
        n_elements=n_elements,
        block_size=BLOCK_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
        approximate=approximate
    )
    
    # Compute standard deviation using PyTorch
    return torch.std(out, dim=dim, keepdim=keepdim, correction=correction)
