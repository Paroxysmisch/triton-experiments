import triton
import triton.language as tl

@triton.jit
def sgd_kernel(
    param_ptr, grad_ptr, momentum_ptr, 
    lr, weight_decay, momentum, dampening, nesterov, maximize,
    numel
):
    pid = tl.program_id(0)
    block_start = pid * numel

    # Loop over elements
    for i in range(block_start, block_start + numel):
        # Load parameter and gradient
        param = tl.load(param_ptr + i)
        grad = tl.load(grad_ptr + i)

        # Apply weight decay
        if weight_decay != 0:
            grad = grad + weight_decay * param

        # Load or initialize momentum buffer
        if momentum != 0:
            if tl.load(momentum_ptr + i) == 0:
                buf = grad
            else:
                buf = tl.load(momentum_ptr + i)
                buf = momentum * buf + (1 - dampening) * grad
                tl.store(momentum_ptr + i, buf)

            # Apply Nesterov momentum
            if nesterov:
                grad = grad + momentum * buf
            else:
                grad = buf

        # Update parameters
        if maximize:
            param = param + lr * grad
        else:
            param = param - lr * grad

        # Store updated parameter
        tl.store(param_ptr + i, param)

import torch

def SGD(params, lr=1e-3, momentum=0, weight_decay=0, dampening=0, nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None):
    # Ensure parameters are in a list
    if not isinstance(params, list):
        params = [params]

    # Initialize momentum buffers
    momentum_buffers = [torch.zeros_like(p) for p in params]

    # Launch Triton kernel for each parameter tensor
    for param, grad, momentum_buffer in zip(params, [p.grad for p in params], momentum_buffers):
        numel = param.numel()
        sgd_kernel[(1,)](
            param, grad, momentum_buffer, 
            lr, weight_decay, momentum, dampening, nesterov, maximize,
            numel
        )

# Example usage
# model_params = [torch.randn(10, requires_grad=True) for _ in range(2)]
# optimizer = SGD(model_params, lr=0.01, momentum=0.9, nesterov=True)
