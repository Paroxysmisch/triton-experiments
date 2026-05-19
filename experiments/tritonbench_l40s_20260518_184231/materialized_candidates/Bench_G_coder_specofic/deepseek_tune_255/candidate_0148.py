import torch
import triton
import triton.language as tl

@triton.jit
def mul2_kernel(
    in_ptr0,
    out_ptr0,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    output = x * 2
    tl.store(out_ptr0 + offsets, output, mask=mask)

def triton_mul2(x: torch.Tensor):
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mul2_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out

@triton.jit
def mul2_inplace_kernel(
    inout_ptr0,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(inout_ptr0 + offsets, mask=mask)
    output = x * 2
    tl.store(inout_ptr0 + offsets, output, mask=mask)

def triton_mul2_inplace(x: torch.Tensor):
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mul2_inplace_kernel[grid](x, n_elements, BLOCK_SIZE=1024)
    return x
