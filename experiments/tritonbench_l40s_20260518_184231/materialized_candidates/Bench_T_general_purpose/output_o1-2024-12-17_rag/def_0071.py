import torch
import triton
import triton.language as tl

@triton.jit
def sgd_kernel(
    p_ptr,            # pointer to parameter tensor
    g_ptr,            # pointer to gradient tensor
    m_ptr,            # pointer to momentum buffer
    lr,               # learning rate
    momentum,         # momentum coefficient
    weight_decay,     # weight decay
    dampening,        # dampening for momentum
    nesterov,         # nesterov boolean flag (0 or 1)
    maximize,         # maximize boolean flag (0 or 1)
    n_elements,       # number of elements
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load parameters and gradients
    p = tl.load(p_ptr + offsets, mask=mask)
    g = tl.load(g_ptr + offsets, mask=mask)

    # Apply weight decay
    if weight_decay != 0.0:
        p_decay = p * weight_decay
        g = g + p_decay

    # Load or initialize momentum buffer
    m = tl.load(m_ptr + offsets, mask=mask)

    # If momentum != 0, compute momentum update
    use_momentum = momentum != 0.0
    if use_momentum:
        # b_t = momentum * b_{t-1} + (1 - dampening) * g
        m_new = momentum * m + (1.0 - dampening) * g
        if nesterov != 0:
            # g = g + momentum * b_t
            g = g + momentum * m_new
        else:
            # g = b_t
            g = m_new
    else:
        m_new = m  # no-op if momentum = 0

    # Maximize or minimize
    if maximize != 0:
        # p = p + lr * g
        p_new = p + lr * g
    else:
        # p = p - lr * g
        p_new = p - lr * g

    # Store updated parameter and momentum
    tl.store(p_ptr + offsets, p_new, mask=mask)
    if use_momentum:
        tl.store(m_ptr + offsets, m_new, mask=mask)


def SGD(
    params,
    lr=1e-3,
    momentum=0,
    weight_decay=0,
    dampening=0,
    nesterov=False,
    maximize=False,
    foreach=None,
    differentiable=False,
    fused=None
):
    """
    Implements SGD algorithm in Triton with optional momentum, weight decay, dampening,
    Nesterov momentum, and maximize mode.
    """
    # For simplicity, handle a list of (param, grad, momentum_buffer) tuples
    # or single (param, grad) with no momentum.
    # This example shows basic per-parameter execution for demonstration.
    if not isinstance(params, list):
        params = [params]

    for group in params:
        # Each 'group' is expected to be a dictionary-like or tuple-like structure:
        # group["params"], group["grads"], group["momentum_buffer"] ...
        # Here, we'll assume direct usage of param, grad, momentum tensors
        # in a tuple of (param_tensor, grad_tensor, momentum_tensor).
        if isinstance(group, dict):
            p_t = group["params"]
            g_t = group["grads"]
            if "momentum_buffer" not in group or group["momentum_buffer"] is None:
                # Initialize momentum buffer if needed
                if momentum != 0:
                    group["momentum_buffer"] = torch.zeros_like(p_t)
                m_t = group["momentum_buffer"]
            else:
                m_t = group["momentum_buffer"]
        else:
            # tuple-like (p_t, g_t, m_t)
            if len(group) == 2:
                p_t, g_t = group
                m_t = None
            else:
                p_t, g_t, m_t = group

        if m_t is None and momentum != 0:
            m_t = torch.zeros_like(p_t)

        assert p_t.is_cuda and g_t.is_cuda, "Tensors must be on CUDA."
        if momentum != 0:
            assert m_t.is_cuda, "Momentum buffer must be on CUDA."

        # Prepare grid
        n_elements = p_t.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        # Convert booleans to int for kernel
        nesterov_flag = 1 if nesterov else 0
        maximize_flag = 1 if maximize else 0

        # Launch kernel
        BLOCK_SIZE = 1024
        sgd_kernel[grid](
            p_t,
            g_t,
            m_t if m_t is not None else p_t,  # dummy pointer if momentum=0
            lr,
            momentum,
            weight_decay,
            dampening,
            nesterov_flag,
            maximize_flag,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
