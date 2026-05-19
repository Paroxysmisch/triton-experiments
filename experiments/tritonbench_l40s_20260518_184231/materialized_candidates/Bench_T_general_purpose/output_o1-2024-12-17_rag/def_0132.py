import triton
import triton.language as tl
import torch
import math

@triton.jit
def _mul_sub_kernel_tt(
    input_ptr, other_mul_ptr, other_sub_ptr, alpha, out_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    inp = tl.load(input_ptr + offsets, mask=mask)
    omul = tl.load(other_mul_ptr + offsets, mask=mask)
    osub = tl.load(other_sub_ptr + offsets, mask=mask)

    out_val = (inp * omul) - alpha * osub
    tl.store(out_ptr + offsets, out_val, mask=mask)

@triton.jit
def _mul_sub_kernel_ts(
    input_ptr, other_mul_ptr, other_sub, alpha, out_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    inp = tl.load(input_ptr + offsets, mask=mask)
    omul = tl.load(other_mul_ptr + offsets, mask=mask)

    out_val = (inp * omul) - alpha * other_sub
    tl.store(out_ptr + offsets, out_val, mask=mask)

@triton.jit
def _mul_sub_kernel_st(
    input_ptr, other_mul, other_sub_ptr, alpha, out_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    inp = tl.load(input_ptr + offsets, mask=mask)
    osub = tl.load(other_sub_ptr + offsets, mask=mask)

    out_val = (inp * other_mul) - alpha * osub
    tl.store(out_ptr + offsets, out_val, mask=mask)

@triton.jit
def _mul_sub_kernel_ss(
    input_ptr, other_mul, other_sub, alpha, out_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    inp = tl.load(input_ptr + offsets, mask=mask)
    out_val = (inp * other_mul) - alpha * other_sub
    tl.store(out_ptr + offsets, out_val, mask=mask)

def mul_sub(input, other_mul, other_sub, alpha=1, out=None):
    if out is None:
        out = torch.empty_like(input)
    n_elements = input.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    
    alpha_val = float(alpha)

    is_omul_t = isinstance(other_mul, torch.Tensor)
    is_osub_t = isinstance(other_sub, torch.Tensor)

    if is_omul_t and is_osub_t:
        _mul_sub_kernel_tt[(grid_size,)](
            input, other_mul, other_sub, alpha_val, out,
            n_elements, block_size
        )
    elif is_omul_t and not is_osub_t:
        _mul_sub_kernel_ts[(grid_size,)](
            input, other_mul, other_sub, alpha_val, out,
            n_elements, block_size
        )
    elif not is_omul_t and is_osub_t:
        _mul_sub_kernel_st[(grid_size,)](
            input, other_mul, other_sub, alpha_val, out,
            n_elements, block_size
        )
    else:
        _mul_sub_kernel_ss[(grid_size,)](
            input, other_mul, other_sub, alpha_val, out,
            n_elements, block_size
        )

    return out
