import torch
import triton
import triton.language as tl

@triton.jit
def update_fn_kernel(
    # Pointers to tensors
    p_ptr,
    grad_ptr,
    exp_avg_ptr,
    # Number of elements
    n_elements,
    # Hyperparameters
    lr,
    wd,
    beta1,
    beta2,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)

    # Apply step weight decay
    p = p * (1 - lr * wd)

    # Compute difference between exp_avg and grad
    diff = exp_avg - grad

    # Update parameter using momentum term beta1
    step = beta1 * diff
    p = p - lr * step

    # Apply sign-based adjustment if there's a change (step != 0)
    step_non_zero = step != 0.0
    sign_step = tl.where(step > 0, 1.0, -1.0)
    # Adjustment: example using learning rate and sign (modify as needed)
    adjustment = lr * sign_step
    p = tl.where(step_non_zero, p - adjustment, p)

    # Update exponential moving average with beta2
    exp_avg = beta2 * exp_avg + (1 - beta2) * grad

    # Store updated values
    tl.store(p_ptr + offsets, p, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)

def update_fn(
    p: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    lr: float,
    wd: float,
    beta1: float,
    beta2: float,
    BLOCK_SIZE: int = 1024,
):
    assert p.is_cuda and grad.is_cuda and exp_avg.is_cuda, "Inputs must be CUDA tensors"
    assert p.is_contiguous() and grad.is_contiguous() and exp_avg.is_contiguous(), "Inputs must be contiguous"
    n_elements = p.numel()
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)
    update_fn_kernel[grid](
        p, grad, exp_avg,
        n_elements,
        lr, wd, beta1, beta2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
