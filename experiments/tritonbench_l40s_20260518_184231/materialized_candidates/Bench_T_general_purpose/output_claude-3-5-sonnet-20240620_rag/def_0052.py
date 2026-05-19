import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional, Tuple, Union
from triton.runtime.jit import get_cuda_stream

@triton.jit
def sum_std_kernel(
    input_ptr,
    output_ptr,
    stride,
    n_elements,
    correction,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask)
    x = x.to(tl.float32)
    
    # Compute sum
    block_sum = tl.sum(x, axis=0)
    
    # Compute mean
    mean = block_sum / n_elements
    
    # Compute variance
    diff = x - mean
    sq_diff = diff * diff
    variance = tl.sum(sq_diff, axis=0)
    
    # Apply Bessel's correction
    n_minus_correction = tl.maximum(0.0, float(n_elements - correction))
    variance = variance / n_minus_correction
    
    # Compute standard deviation
    std = tl.sqrt(variance)
    
    # Store results (sum and std)
    if pid == 0:
        tl.store(output_ptr, block_sum)
        tl.store(output_ptr + 1, std)

@torch.inference_mode()
def sum_std(
    input: Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    dtype: Optional[torch.dtype] = None,
    correction: int = 1,
    out: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """
    Computes the sum and standard deviation of elements in the input tensor.
    
    Args:
        input (Tensor): The input tensor
        dim (int or tuple of ints, optional): Dimension(s) to reduce. If None, reduces all dimensions
        keepdim (bool): Whether to keep the reduced dimensions
        dtype (torch.dtype, optional): Desired data type of output
        correction (int): Bessel's correction factor (default: 1)
        out (Tensor, optional): Output tensor
        
    Returns:
        tuple: (sum, std) tensors
    """
    # Handle dtype
    if dtype is None:
        dtype = input.dtype
    input = input.to(dtype)
    
    # If dim is None, flatten the input
    if dim is None:
        input = input.flatten()
        dim = 0
    
    # Convert single dim to tuple
    if isinstance(dim, int):
        dim = (dim,)
    
    # Calculate number of elements to reduce
    n_elements = 1
    for d in dim:
        n_elements *= input.shape[d]
    
    # Prepare output tensor
    if out is None:
        output_shape = list(input.shape)
        if not keepdim:
            for d in sorted(dim, reverse=True):
                output_shape.pop(d)
        out = torch.empty(output_shape, dtype=dtype, device=input.device)
    
    # Initialize result tensor for sum and std
    result = torch.empty(2, dtype=torch.float32, device=input.device)
    
    # Launch kernel
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 1024))
    grid = (1,)
    
    sum_std_kernel[grid](
        input.data_ptr(),
        result.data_ptr(),
        input.stride(0),
        n_elements,
        correction,
        BLOCK_SIZE,
        num_warps=4,
    )
    
    sum_val, std_val = result[0], result[1]
    
    # Reshape output if keepdim is True
    if keepdim:
        output_shape = list(input.shape)
        for d in dim:
            output_shape[d] = 1
        sum_val = sum_val.view(output_shape)
        std_val = std_val.view(output_shape)
    
    return sum_val, std_val
