import torch
import triton
import triton.language as tl

@triton.jit
def update_fn_kernel(
    p_ptr,
    grad_ptr,
    exp_avg_ptr,
    n_elements,
    lr,
    wd,
    beta1,
    beta2,
    TID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    p_ptr += offsets
    grad_ptr += offsets
    exp_avg_ptr += offsets

    p = tl.load(p_ptr, mask=mask)
    grad = tl.load(grad_ptr, mask=mask)
    exp_avg = tl.load(exp_avg_ptr, mask=mask)

    p = p * (1 - lr * wd)

    diff = exp_avg - grad
    update = diff * beta1

    p_new = p + update

    sign_bit = tl.where(p_new > 0, 0, -0.)
    sign_bit = tl.sign(p_new)
    p_new = p_new + sign_bit * 1e-7

    tl.store(p_ptr, p_new, mask=mask)

    exp_avg_new = exp_avg * beta2 + grad
    tl.store(exp_avg_ptr, exp_avg_new, mask=mask)


def update_fn(
    p: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    lr: float,
    wd: float,
    beta1: float,
    beta2: float,
):
    assert all([t.is_cuda for t in (p, grad, exp_avg)]), "All tensors must be CUDA"
    n_elements = p.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    update_fn_kernel[grid](
        p,
        grad,
        exp_avg,
        n_elements,
        lr,
        wd,
        beta1,
        beta2,
    )

    return p, exp_avg
