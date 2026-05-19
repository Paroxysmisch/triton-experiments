import triton
import triton.language as tl
import torch
import math

@triton.jit
def sqrt_tanh_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input element and convert to float32 for computation
    input_val = tl.load(input_ptr + offsets, mask=mask)
    input_float = input_val.to(tl.float32)
    
    # Compute sqrt and tanh
    sqrt_val = tl.sqrt(input_float)
    tanh_val = tl.tanh(sqrt_val)
    
    # Convert the result to the output tensor's data type
    output_dtype = output_ptr.dtype.element_ty
    output_val = tanh_val.to(output_dtype)
    
    # Store the result
    tl.store(output_ptr + offsets, output_val, mask=mask)

def sqrt_tanh(input, out=None) -> torch.Tensor:
    # Ensure input is a tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    
    # Make the input contiguous
    input = input.contiguous()
    
    # Handle the output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        if not out.is_contiguous():
            raise ValueError("out tensor must be contiguous")
        if out.shape != input.shape:
            raise ValueError("out tensor must have the same shape as input")
    
    n_elements = input.numel()
    if n_elements == 0:
        return out  # No elements to process
    
    # Compute block and grid sizes
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    
    # Launch kernel
    sqrt_tanh_kernel[(grid_size, 1, 1)](
        input, out, n_elements, BLOCK_SIZE=block_size
    )
    
    return out
