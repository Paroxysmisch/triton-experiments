import torch
import triton
import triton.language as tl

@triton.jit
def zeta_kernel(x_ptr, q_ptr, zeta_ptr, n_elements,
                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    q = tl.load(q_ptr + offsets, mask=mask)

    sum_ = 0.0
    k = 0
    cond = True
    while cond:
        k_ = k * 1.0
        q_ = q + 1
        x_ = x
        power = tl.exp(tl.math.log(1.0 + (1.0 / ((k_ + q_) ** x_))) * x_)
        sum_ += power
        k += 1
        if k >= BLOCK_SIZE:
            k = 0
            cond = False

    zeta = sum_
    tl.store(zeta_ptr + offsets, zeta, mask=mask)


def zeta(input: torch.Tensor, other: torch.Tensor, *, out=None):
    if out is None:
        out = torch.empty_like(input)

    assert input.is_cuda and other.is_cuda and out.is_cuda
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    zeta_kernel[grid](input, other, out, n_elements, BLOCK_SIZE=1024)
    return out
