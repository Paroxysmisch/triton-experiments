import torch
import triton
import triton.language as tl

@triton.jit
def update_fn_kernel(
    p_ptr,
    grad_ptr,
    exp_avg_ptr,
    lr,
    wd,
    beta1,
    beta2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Offsetted pointers
    offset_p_ptr = p_ptr + offsets
    offset_grad_ptr = grad_ptr + offsets
    offset_exp_avg_ptr = exp_avg_ptr + offsets

    # Load
    p = tl.load(offset_p_ptr, mask=mask)
    grad = tl.load(offset_grad_ptr, mask=mask)
    exp_avg = tl.load(offset_exp_avg_ptr, mask=mask)

    # Step weight decay
    p = p * (1 - lr * wd)

    # Diff between exp_avg and grad
    diff = exp_avg - grad

    # Update p
    update_p = diff * beta1 + grad
    p = p + update_p
    tl.store(offset_p_ptr, p, mask=mask)

    # Sign
    # NOTE: We can't use `torch.sign` because AFAIK it's not supported by triton
    # So we emulate it with this
    change = update_p != 0
    p_positive = p >= 0
    sign = p_positive ^ change
    neg_update = tl.where(sign, -update_p, update_p)
    p = p - neg_update
    tl.store(offset_p_ptr, p, mask=mask)

    # Decay exp_avg
    exp_avg = diff * beta2 + grad
    tl.store(offset_exp_avg_ptr, exp_avg, mask=mask)


def update_fn(
    p: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    lr: float,
    wd: float,
    beta1: float,
    beta2: float,
) -> None:
    assert p.is_cuda and grad.is_cuda and exp_avg.is_cuda

    N = p.numel()
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    update_fn_kernel[grid](
        p,
        grad,
        exp_avg,
        lr,
        wd,
        beta1,
        beta2,
        N,
        BLOCK_SIZE=1024,
    )
