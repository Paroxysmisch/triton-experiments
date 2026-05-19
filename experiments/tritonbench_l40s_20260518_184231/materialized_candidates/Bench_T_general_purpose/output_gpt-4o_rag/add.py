import triton
import triton.language as tl
import torch
import math

# Kernel for adding a scaled tensor or number to an input tensor
@triton.jit
def add_scaled_kernel(input_ptr, other_ptr, alpha, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    A kernel to add a scaled version of 'other' to 'input' and store the result in 'out'.
    
    Parameters:
    - input_ptr: Pointer to the input tensor.
    - other_ptr: Pointer to the other tensor (or None if 'other' is a scalar).
    - alpha: Scalar multiplier for 'other'.
    - out_ptr: Pointer to the output tensor.
    - n_elements: Total number of elements in the output tensor.
    - BLOCK_SIZE: The block size for Triton kernel execution.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input_val = tl.load(input_ptr + offsets, mask=mask)
    
    if other_ptr is not None:
        other_val = tl.load(other_ptr + offsets, mask=mask)
    else:
        other_val = tl.broadcast(tl.load(other_ptr), input_val.shape)
    
    result = input_val + alpha * other_val
    tl.store(out_ptr + offsets, result, mask=mask)

# Wrapper function to handle the addition
def add(input, other, *, alpha=1, out=None):
    """
    A wrapper function to add 'other', scaled by 'alpha', to 'input'.
    
    Parameters:
    - input: The input tensor.
    - other: The tensor or number to add to input.
    - alpha: The multiplier for other.
    - out: The output tensor (optional).
    
    Returns:
    - The resulting tensor after the addition.
    """
    # Determine the output shape and allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    
    # Broadcast 'other' to match the shape of 'input' if necessary
    if isinstance(other, torch.Tensor):
        other_broadcasted = torch.broadcast_to(other, input.shape)
        other_ptr = other_broadcasted
    else:
        other_ptr = None

    # Calculate number of elements and block size
    n_elements = out.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)

    # Call the kernel
    add_scaled_kernel[(grid_size,)](input, other_ptr, alpha, out, n_elements, block_size)
    
    return out
