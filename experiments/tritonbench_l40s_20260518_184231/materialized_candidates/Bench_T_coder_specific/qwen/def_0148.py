import triton
import triton.language as tl

@triton.jit
def adam_kernel(
    params_ptr,
    grad_ptr,
    m_ptr,
    v_ptr,
    m_hat_ptr,
    v_hat_ptr,
    step_ptr,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    n,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * block_size + tl.arange(0, block_size)

    # Load parameters, gradients, and state variables
    params = tl.load(params_ptr + offsets)
    grads = tl.load(grad_ptr + offsets)
    ms = tl.load(m_ptr + offsets)
    vs = tl.load(v_ptr + offsets)

    # Update biased first moment estimate
    ms_new = ms * beta1 + grads * (1 - beta1)

    # Update biased second raw moment estimate
    vs_new = vs * beta2 + grads * grads * (1 - beta2)

    # Compute bias-corrected first moment estimate
    ms_hat = ms_new / (1 - beta1 ** step_ptr[0])

    # Compute bias-corrected second raw moment estimate
    vs_hat = vs_new / (1 - beta2 ** step_ptr[0])

    # Apply weight decay
    params_decayed = params * (1 - lr * weight_decay)

    # Update parameters
    new_params = params_decayed - lr * ms_hat / (tl.sqrt(vs_hat) + eps)

    # Store updated values back to global memory
    tl.store(params_ptr + offsets, new_params)
    tl.store(m_ptr + offsets, ms_new)
    tl.store(v_ptr + offsets, vs_new)
