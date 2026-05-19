import torch
import triton
import triton.language as tl

@triton.jit
def sgd_kernel(
    params_ptr, 
    grads_ptr, 
    momentum_ptr, 
    lr, 
    weight_decay, 
    momentum, 
    dampening, 
    nesterov, 
    maximize, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load parameters, gradients, and momentum
    params = tl.load(params_ptr + offsets, mask=mask)
    grads = tl.load(grads_ptr + offsets, mask=mask)
    momentum_buffer = tl.load(momentum_ptr + offsets, mask=mask)

    # Apply weight decay
    if weight_decay != 0:
        grads = grads + weight_decay * params

    # Update momentum buffer
    if momentum != 0:
        if dampening != 0:
            momentum_buffer = momentum * momentum_buffer + (1 - dampening) * grads
        else:
            momentum_buffer = momentum * momentum_buffer + grads
        
        if nesterov:
            grads = grads + momentum * momentum_buffer
        else:
            grads = momentum_buffer

    # Update parameters
    if maximize:
        params = params + lr * grads
    else:
        params = params - lr * grads

    # Store updated parameters and momentum buffer
    tl.store(params_ptr + offsets, params, mask=mask)
    tl.store(momentum_ptr + offsets, momentum_buffer, mask=mask)

def SGD(params, lr=1e-3, momentum=0, weight_decay=0, dampening=0, nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None):
    assert all([p.is_cuda for p in params]), "All parameters must be CUDA tensors."
    n_elements = params[0].numel()
    momentum_buffers = [torch.zeros_like(p) for p in params]

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    for p, g, m in zip(params, grads, momentum_buffers):
        sgd_kernel[grid](
            p, 
            g, 
            m, 
            lr, 
            weight_decay, 
            momentum, 
            dampening, 
            nesterov, 
            maximize, 
            n_elements
        )
