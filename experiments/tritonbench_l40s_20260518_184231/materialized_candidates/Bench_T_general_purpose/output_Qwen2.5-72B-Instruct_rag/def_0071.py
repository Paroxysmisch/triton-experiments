import triton
import triton.language as tl

@triton.jit
def sgd_kernel(
    p_ptr,  # Pointer to parameters
    grad_ptr,  # Pointer to gradients
    exp_avg_ptr,  # Pointer to momentum buffer
    lr,  # Learning rate
    momentum,  # Momentum coefficient
    weight_decay,  # Weight decay coefficient
    dampening,  # Dampening coefficient
    nesterov,  # Nesterov momentum flag
    maximize,  # Maximize or minimize objective function
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr  # Number of threads per block
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load parameters, gradients, and momentum buffer
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)

    # Apply weight decay
    if weight_decay != 0:
        grad = grad + weight_decay * p

    # Update momentum buffer
    if momentum != 0:
        if pid == 0:  # First step, initialize momentum buffer
            exp_avg = grad
        else:
            exp_avg = momentum * exp_avg + (1 - dampening) * grad

        if nesterov:
            grad = grad + momentum * exp_avg
        else:
            grad = exp_avg

    # Update parameters
    if maximize:
        p = p + lr * grad
    else:
        p = p - lr * grad

    # Store updated parameters and momentum buffer
    tl.store(p_ptr + offsets, p, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)

import torch

def SGD(params, lr=1e-3, momentum=0, weight_decay=0, dampening=0, nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None):
    if foreach is not None or differentiable or fused is not None:
        raise NotImplementedError("foreach, differentiable, and fused options are not supported in this implementation.")

    for param in params:
        if param.grad is None:
            continue

        # Ensure tensors are on the same device
        assert param.is_cuda and param.grad.is_cuda

        # Initialize momentum buffer if it doesn't exist
        if 'exp_avg' not in param.state:
            param.state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)

        # Get pointers to the data
        p_ptr = param.data_ptr()
        grad_ptr = param.grad.data_ptr()
        exp_avg_ptr = param.state['exp_avg'].data_ptr()

        # Number of elements
        n_elements = param.numel()

        # Set up grid and block dimensions
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

        # Launch the kernel
        sgd_kernel[grid](
            p_ptr,
            grad_ptr,
            exp_avg_ptr,
            lr,
            momentum,
            weight_decay,
            dampening,
            nesterov,
            maximize,
            n_elements,
            BLOCK_SIZE=1024
        )

    return params
