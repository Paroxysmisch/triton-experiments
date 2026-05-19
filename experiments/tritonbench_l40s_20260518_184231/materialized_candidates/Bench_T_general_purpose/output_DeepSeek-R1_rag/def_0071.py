import torch
import triton
import triton.language as tl

@triton.jit
def sgd_kernel(
    p_ptr,
    grad_ptr,
    buf_ptr,
    lr,
    wd,
    momentum,
    dampening,
    nesterov,
    maximize,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load parameters
    p = tl.load(p_ptr + offsets, mask=mask)
    g = tl.load(grad_ptr + offsets, mask=mask)
    b = tl.load(buf_ptr + offsets, mask=mask)

    # Apply weight decay
    if wd != 0.0:
        g += wd * p

    # Update momentum buffer
    if momentum != 0.0:
        b_prev = b
        b = momentum * b_prev + (1 - dampening) * g
    else:
        b = g  # If momentum is 0, buffer is gradient (but effectively unused)

    # Determine update direction
    if momentum != 0.0 and nesterov:
        update = g + momentum * b
    elif momentum != 0.0:
        update = b
    else:
        update = g

    # Apply parameter update
    if maximize:
        p += lr * update
    else:
        p -= lr * update

    # Store updated values
    tl.store(p_ptr + offsets, p, mask=mask)
    if momentum != 0.0:
        tl.store(buf_ptr + offsets, b, mask=mask)

def SGD(params, lr=1e-3, momentum=0, weight_decay=0, dampening=0, nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None):
    for param in params:
        if param.grad is None:
            continue
        grad = param.grad.data
        if grad.is_sparse:
            raise RuntimeError("SGD does not support sparse gradients")

        # Initialize momentum buffer if needed
        if momentum != 0:
            param_state = param
            if not hasattr(param_state, 'momentum_buffer'):
                buf = torch.zeros_like(param.data)
                param_state.momentum_buffer = buf
            else:
                buf = param_state.momentum_buffer
        else:
            buf = torch.empty(0, device=param.data.device)  # Dummy buffer

        # Launch kernel
        n_elements = param.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        sgd_kernel[grid](
            param.data,
            grad,
            buf if momentum != 0 else param.data,  # Avoid passing dummy buffer
            lr,
            weight_decay,
            momentum,
            dampening,
            nesterov,
            maximize,
            n_elements,
            BLOCK_SIZE=1024,
        )
