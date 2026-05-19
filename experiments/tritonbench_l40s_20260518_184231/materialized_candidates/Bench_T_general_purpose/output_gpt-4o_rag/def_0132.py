import triton
import triton.language as tl
import torch
import math

# Kernel for element-wise operation with tensor inputs
@triton.jit
def mul_sub_tensor_kernel(input_ptr, other_mul_ptr, other_sub_ptr, out_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input = tl.load(input_ptr + offsets, mask=mask)
    other_mul = tl.load(other_mul_ptr + offsets, mask=mask)
    other_sub = tl.load(other_sub_ptr + offsets, mask=mask)

    result = (input * other_mul) - (alpha * other_sub)
    tl.store(out_ptr + offsets, result, mask=mask)

# Kernel for element-wise operation with scalar inputs
@triton.jit
def mul_sub_scalar_kernel(input_ptr, other_mul, other_sub, out_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input = tl.load(input_ptr + offsets, mask=mask)

    result = (input * other_mul) - (alpha * other_sub)
    tl.store(out_ptr + offsets, result, mask=mask)

# Wrapper function
def mul_sub(input, other_mul, other_sub, alpha=1, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    n_elements = input.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)

    if isinstance(other_mul, torch.Tensor) and isinstance(other_sub, torch.Tensor):
        # Call kernel for tensor-tensor operations
        mul_sub_tensor_kernel[(grid_size,)](input, other_mul, other_sub, out, alpha, n_elements, block_size)
    else:
        # Call kernel for scalar operations
        if isinstance(other_mul, torch.Tensor):
            other_mul = other_mul.item()
        if isinstance(other_sub, torch.Tensor):
            other_sub = other_sub.item()
        mul_sub_scalar_kernel[(grid_size,)](input, other_mul, other_sub, out, alpha, n_elements, block_size)
    
    return out
