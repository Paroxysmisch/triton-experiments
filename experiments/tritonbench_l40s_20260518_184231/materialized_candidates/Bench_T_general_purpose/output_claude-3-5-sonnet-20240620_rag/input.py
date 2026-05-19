import torch
import triton
import triton.language as tl
from typing import Optional, Union, Tuple
from numbers import Number

@triton.jit
def mean_kernel(
    output_ptr,
    input_ptr,
    stride_dim,
    reduce_dim_size,
    other_dims_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate start position for this program
    base_idx = pid * BLOCK_SIZE
    
    # Create offsets for the reduction dimension
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < reduce_dim_size
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Calculate input pointers
    input_ptrs = input_ptr + base_idx * stride_dim + offsets
    
    # Load and accumulate values
    x = tl.load(input_ptrs, mask=mask, other=0.0)
    acc = tl.sum(x, axis=0) / reduce_dim_size
    
    # Store result
    output_ptr = output_ptr + pid
    tl.store(output_ptr, acc)

def mean(
    input: torch.Tensor,
    dim: Union[int, Tuple[int, ...]] = None,
    keepdim: bool = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes the mean of the input tensor along the specified dimension(s).
    
    Args:
        input: Input tensor
        dim: Dimension or dimensions to reduce over
        keepdim: Whether to keep the reduced dimensions
        dtype: Optional dtype for the output tensor
        out: Optional output tensor
    
    Returns:
        Tensor: The mean values along the specified dimension(s)
    """
    # Handle input validation and preparation
    if dtype is None:
        dtype = input.dtype
    
    # Convert input to specified dtype
    input = input.to(dtype)
    
    # Handle dim=None case (reduce over all dimensions)
    if dim is None:
        dim = tuple(range(input.dim()))
    elif isinstance(dim, Number):
        dim = (dim,)
    
    # Normalize negative dimensions
    dim = tuple(d if d >= 0 else d + input.dim() for d in dim)
    
    # Calculate output shape
    output_shape = list(input.shape)
    reduce_size = 1
    for d in dim:
        reduce_size *= input.shape[d]
        output_shape[d] = 1 if keepdim else None
    output_shape = [s for s in output_shape if s is not None]
    
    # Reshape input to combine all reduction dimensions
    other_dims = [i for i in range(input.dim()) if i not in dim]
    permutation = list(dim) + other_dims
    input = input.permute(permutation)
    other_dims_size = input.numel() // reduce_size
    
    # Reshape for reduction
    input = input.reshape(reduce_size, -1)
    
    # Prepare output tensor
    if out is None:
        output = torch.empty(other_dims_size, device=input.device, dtype=dtype)
    else:
        output = out.reshape(other_dims_size)
    
    # Launch kernel
    BLOCK_SIZE = triton.next_power_of_2(min(reduce_size, 1024))
    grid = (other_dims_size,)
    
    mean_kernel[grid](
        output,
        input,
        input.stride(0),
        reduce_size,
        other_dims_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape output to final shape
    if keepdim:
        output = output.reshape(output_shape)
    
    return output if out is None else out
