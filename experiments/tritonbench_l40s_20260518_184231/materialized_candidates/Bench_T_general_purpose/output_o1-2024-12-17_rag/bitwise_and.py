import triton
import triton.language as tl
import torch
import math

@triton.jit
def _bitwise_and_tensor_kernel(A_ptr, B_ptr, C_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    A = tl.load(A_ptr + offsets, mask=mask)
    B = tl.load(B_ptr + offsets, mask=mask)
    
    C = A & B
    
    tl.store(C_ptr + offsets, C, mask=mask)

@triton.jit
def _bitwise_and_scalar_kernel(A_ptr, B: tl.constexpr, C_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    A = tl.load(A_ptr + offsets, mask=mask)
    
    C = A & B
    
    tl.store(C_ptr + offsets, C, mask=mask)

def bitwise_and(input, other, *, out=None):
    if not (input.dtype.is_floating_point is False and (isinstance(other, (int, bool, torch.Tensor)))):
        raise TypeError("Both input and other must be integral or boolean types.")

    if out is None:
        out = torch.empty_like(input)

    n_elements = input.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)

    if isinstance(other, torch.Tensor):
        if not (other.dtype.is_floating_point is False):
            raise TypeError("Both input and other must be integral or boolean types.")
        _bitwise_and_tensor_kernel[(grid_size,)](
            input, other, out, n_elements, block_size
        )
    else:
        _bitwise_and_scalar_kernel[(grid_size,)](
            input, other, out, n_elements, block_size
        )
    
    return out
