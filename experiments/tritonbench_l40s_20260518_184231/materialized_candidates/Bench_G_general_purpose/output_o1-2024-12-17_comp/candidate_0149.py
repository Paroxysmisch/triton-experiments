import triton
import triton.language as tl
import math

@triton.jit
def update_fn_kernel(
    p_ptr, grad_ptr, exp_avg_ptr,
    lr, wd, beta1, beta2,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0)

    # Step weight decay
    p = p * (1 - lr * wd)

    # Sign-based momentum update
    diff = exp_avg - grad
    sign_diff = tl.where(diff > 0.0, 1.0, tl.where(diff < 0.0, -1.0, 0.0))
    update_val = lr * beta1 * sign_diff
    p += update_val

    # Update exponential moving average
    exp_avg = exp_avg * beta2 + grad * (1 - beta2)

    tl.store(p_ptr + offsets, p, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)

def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2, block_size=1024):
    assert p.is_cuda and grad.is_cuda and exp_avg.is_cuda, "All tensors must be CUDA tensors."
    n_elements = p.numel()
    grid = lambda meta: (math.ceil(n_elements / meta['BLOCK_SIZE']),)
    update_fn_kernel[grid](p, grad, exp_avg, lr, wd, beta1, beta2, n_elements, BLOCK_SIZE=block_size)
