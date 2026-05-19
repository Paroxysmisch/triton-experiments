import triton
import triton.language as tl

@triton.jit
def adam_kernel(
    params_ptr, grads_ptr, m_ptr, v_ptr, 
    lr, beta1, beta2, eps, weight_decay, 
    step, n_params,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_params

    # Load parameters, gradients, m, and v
    params = tl.load(params_ptr + offsets, mask=mask)
    grads = tl.load(grads_ptr + offsets, mask=mask)
    m = tl.load(m_ptr + offsets, mask=mask)
    v = tl.load(v_ptr + offsets, mask=mask)

    # Apply weight decay if needed
    if weight_decay != 0:
        grads += weight_decay * params

    # Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * grads

    # Update biased second raw moment estimate
    v = beta2 * v + (1 - beta2) * grads * grads

    # Compute bias-corrected first moment estimate
    m_hat = m / (1 - beta1 ** step)

    # Compute bias-corrected second raw moment estimate
    v_hat = v / (1 - beta2 ** step)

    # Update parameters
    denom = tl.sqrt(v_hat) + eps
    params -= lr * m_hat / denom

    # Store updated parameters, m, and v
    tl.store(params_ptr + offsets, params, mask=mask)
    tl.store(m_ptr + offsets, m, mask=mask)
    tl.store(v_ptr + offsets, v, mask=mask)


class Optimizer:
    def __init__(self, params):
        self.params = params

def Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None) -> Optimizer:
    import torch

    # Initialize state
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    step = 1

    def step_fn():
        nonlocal step
        for i, param in enumerate(params):
            grad = param.grad
            if grad is None:
                continue

            # If maximize is True, negate the gradient
            if maximize:
                grad = -grad

            # Prepare pointers for Triton kernel
            params_ptr = param.data_ptr()
            grads_ptr = grad.data_ptr()
            m_ptr = m[i].data_ptr()
            v_ptr = v[i].data_ptr()
            n_params = param.numel()

            # Launch Triton kernel
            grid = lambda meta: (triton.cdiv(n_params, meta['BLOCK_SIZE']),)
            adam_kernel[grid](
                params_ptr, grads_ptr, m_ptr, v_ptr,
                lr, betas[0], betas[1], eps, weight_decay,
                step, n_params,
                BLOCK_SIZE=1024
            )

        step += 1

    optimizer = Optimizer(params)
    optimizer.step = step_fn
    return optimizer
