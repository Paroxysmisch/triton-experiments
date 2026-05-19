import triton
import triton.language as tl

@triton.jit
def sgd_kernel(
    params_ptr,  # pointer to the parameters
    grads_ptr,   # pointer to the gradients
    momentums_ptr,  # pointer to the momentum buffer
    lr,  # learning rate
    momentum,  # momentum coefficient
    weight_decay,  # weight decay coefficient
    dampening,  # dampening for momentum
    nesterov,  # use Nesterov momentum
    maximize,  # maximize or minimize the objective
    param_size,  # size of the parameter array
    BLOCK_SIZE: tl.constexpr  # block size for parallelization
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < param_size

    params = tl.load(params_ptr + offsets, mask=mask)
    grads = tl.load(grads_ptr + offsets, mask=mask)
    momentums = tl.load(momentums_ptr + offsets, mask=mask)

    # Apply weight decay
    if weight_decay != 0:
        grads = grads + weight_decay * params

    # Update momentum
    if momentum != 0:
        if dampening == 0:
            momentums = momentum * momentums + grads
        else:
            momentums = momentum * momentums + (1 - dampening) * grads

    # Apply Nesterov momentum
    if nesterov:
        grads = grads + momentum * momentums
    else:
        grads = momentums

    # Update parameters
    if maximize:
        params = params + lr * grads
    else:
        params = params - lr * grads

    tl.store(params_ptr + offsets, params, mask=mask)
    tl.store(momentums_ptr + offsets, momentums, mask=mask)

import torch
import triton

def SGD(params, lr=1e-3, momentum=0, weight_decay=0, dampening=0, nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None):
    if foreach is not None or differentiable or fused is not None:
        raise NotImplementedError("foreach, differentiable, and fused options are not supported in this implementation.")

    # Convert parameters and gradients to contiguous tensors
    params = [p.contiguous() for p in params]
    grads = [p.grad.contiguous() for p in params]

    # Initialize momentum buffers
    momentums = [torch.zeros_like(p) for p in params]

    # Flatten parameters, gradients, and momentums
    params_flat = torch.cat([p.view(-1) for p in params])
    grads_flat = torch.cat([g.view(-1) for g in grads])
    momentums_flat = torch.cat([m.view(-1) for m in momentums])

    # Determine block size for parallelization
    BLOCK_SIZE = 1024
    grid = (params_flat.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    sgd_kernel[grid, BLOCK_SIZE](
        params_flat, grads_flat, momentums_flat,
        lr, momentum, weight_decay, dampening, nesterov, maximize, params_flat.numel()
    )

    # Copy the updated parameters back to the original tensors
    offset = 0
    for p in params:
        size = p.numel()
        p.copy_(params_flat[offset:offset + size].view_as(p))
        offset += size

    return params
