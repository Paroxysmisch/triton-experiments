import triton
import triton.language as tl

@triton.jit
def adam_kernel(
    params_ptr, grads_ptr, m_ptr, v_ptr, h_ptr,
    lr, beta1, beta2, eps, weight_decay, step,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < params_ptr.shape[0]

    params = tl.load(params_ptr + offsets, mask=mask)
    grads = tl.load(grads_ptr + offsets, mask=mask)
    m = tl.load(m_ptr + offsets, mask=mask)
    v = tl.load(v_ptr + offsets, mask=mask)
    h = tl.load(h_ptr + offsets, mask=mask)

    # Apply weight decay
    grads += weight_decay * params

    # Update first moment estimate
    m_new = beta1 * m + (1 - beta1) * grads

    # Update second raw moment estimate
    v_new = beta2 * v + (1 - beta2) * (grads * grads)

    # Bias correction for first moment
    m_hat = m_new / (1 - beta1 ** step)

    # Bias correction for second raw moment
    v_hat = v_new / (1 - beta2 ** step)

    # Compute the step size
    step_size = lr * m_hat / (tl.sqrt(v_hat) + eps)

    # Update parameters
    params -= step_size

    # Store updated values
    tl.store(params_ptr + offsets, params, mask=mask)
    tl.store(m_ptr + offsets, m_new, mask=mask)
    tl.store(v_ptr + offsets, v_new, mask=mask)
    tl.store(h_ptr + offsets, h, mask=mask)
