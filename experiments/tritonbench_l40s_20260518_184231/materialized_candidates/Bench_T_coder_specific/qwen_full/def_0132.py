import torch
import triton
import triton.language as tl

@triton.jit
def _mul_sub_fwd(
    a_ptr, b_ptr, c_ptr, d_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
    ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    x = a * b
    y = c * d
    out = x - y
    tl.store(out_ptr + offsets, out, mask=mask)

def mul_sub(input, other_mul, other_sub, alpha=1, out=None) -> torch.Tensor:
    assert input.is_contiguous()
    assert other_mul.is_contiguous()
    assert other_sub.is_contiguous()
    if out == None:
        out = torch.empty_like(input)
    else:
        assert out.is_contiguous()
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _mul_sub_fwd[grid](input, other_mul, other_sub, other_sub, out, n_elements, BLOCK_SIZE=1024)
    return out
