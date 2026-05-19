import triton
import triton.language as tl
import torch
import math

@triton.jit
def add_tensor_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    other = tl.load(other_ptr + offsets, mask=mask)
    output = input + alpha * other
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def add_scalar_kernel(
    input_ptr,
    output_ptr,
    other_scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = input + other_scalar
    tl.store(output_ptr + offsets, output, mask=mask)

def add(input, other, *, alpha=1, out=None):
    # Determine the common data type
    dtype = torch.result_type(input, other)
    
    # Promote input and other to the common data type
    input = input.to(dtype)
    if isinstance(other, torch.Tensor):
        other = other.to(dtype)
    
    # Determine if other is a tensor or scalar
    is_other_tensor = isinstance(other, torch.Tensor)
    
    if is_other_tensor:
        # Compute the broadcasted shape
        try:
            broadcasted_shape = torch.broadcast_shapes(input.shape, other.shape)
        except Exception as e:
            raise ValueError(f"Shapes {input.shape} and {other.shape} are not broadcastable") from e
        
        # Expand input and other to the broadcasted shape
        input_expanded = input.expand(broadcasted_shape)
        other_expanded = other.expand(broadcasted_shape)
        
        # Check if they are contiguous; if not, make them contiguous
        if not input_expanded.is_contiguous():
            input_expanded = input_expanded.contiguous()
        if not other_expanded.is_contiguous():
            other_expanded = other_expanded.contiguous()
        
        # Prepare output tensor
        if out is not None:
            if out.shape != broadcasted_shape:
                raise ValueError(f"out shape {out.shape} does not match broadcasted shape {broadcasted_shape}")
            if not out.is_contiguous():
                raise ValueError("out tensor must be contiguous")
            if out.dtype != dtype:
                raise TypeError(f"out tensor dtype {out.dtype} does not match expected dtype {dtype}")
            output = out
        else:
            output = torch.empty_like(input_expanded)
        
        n_elements = output.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
        grid_size = triton.cdiv(n_elements, block_size)
        
        add_tensor_kernel[(grid_size, 1, 1)](
            input_expanded, other_expanded, output, alpha, n_elements, block_size
        )
    else:
        # Handle scalar other
        other_scalar = dtype(other) * alpha
        
        # Prepare output tensor
        if out is not None:
            if out.shape != input.shape:
                raise ValueError(f"out shape {out.shape} does not match input shape {input.shape}")
            if not out.is_contiguous():
                raise ValueError("out tensor must be contiguous")
            if out.dtype != dtype:
                raise TypeError(f"out tensor dtype {out.dtype} does not match expected dtype {dtype}")
            output = out
        else:
            output = torch.empty_like(input)
        
        n_elements = output.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
        grid_size = triton.cdiv(n_elements, block_size)
        
        add_scalar_kernel[(grid_size, 1, 1)](
            input, output, other_scalar, n_elements, block_size
        )
    
    return output
