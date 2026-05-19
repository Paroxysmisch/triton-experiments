import torch
import triton
import triton.language as tl

@triton.jit
def sgd_kernel(
    param_ptr,
    grad_ptr,
    momentum_buffer_ptr,
    n_elements,
    lr,
    momentum,
    weight_decay,
    dampening,
    nesterov,
    maximize,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    param = tl.load(param_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    
    # Apply weight decay
    if weight_decay != 0:
        grad = grad + weight_decay * param
    
    if momentum != 0:
        momentum_buffer = tl.load(momentum_buffer_ptr + offsets, mask=mask)
        momentum_buffer = momentum * momentum_buffer + (1 - dampening) * grad
        
        if nesterov:
            grad = grad + momentum * momentum_buffer
        else:
            grad = momentum_buffer
            
        # Store momentum buffer
        tl.store(momentum_buffer_ptr + offsets, momentum_buffer, mask=mask)
    
    # Update parameters
    if maximize:
        param = param + lr * grad
    else:
        param = param - lr * grad
        
    # Store updated parameters
    tl.store(param_ptr + offsets, param, mask=mask)

def sgd_step_triton(
    params: torch.Tensor,
    grads: torch.Tensor,
    momentum_buffers: torch.Tensor,
    lr: float,
    momentum: float,
    weight_decay: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
):
    assert all(p.is_cuda and g.is_cuda for p, g in zip(params, grads))
    if momentum != 0:
        assert momentum_buffers is not None
        assert all(b.is_cuda for b in momentum_buffers)

    for i, (param, grad) in enumerate(zip(params, grads)):
        n_elements = param.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        momentum_buffer = momentum_buffers[i] if momentum != 0 else None
        momentum_buffer_ptr = momentum_buffer.data_ptr() if momentum_buffer is not None else 0
        
        sgd_kernel[grid](
            param.data_ptr(),
            grad.data_ptr(),
            momentum_buffer_ptr,
            n_elements,
            lr,
            momentum,
            weight_decay,
            dampening,
            nesterov,
            maximize,
            BLOCK_SIZE,
        )

def SGD(params, lr=1e-3, momentum=0, weight_decay=0, dampening=0, 
        nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None):
    """
    Implements stochastic gradient descent with momentum.
    
    Args:
        params: iterable of parameters to optimize
        lr: learning rate
        momentum: momentum factor
        weight_decay: weight decay (L2 penalty)
        dampening: dampening for momentum
        nesterov: enables Nesterov momentum
        maximize: maximize the params based on the objective
        foreach: whether to use a for-each implementation
        differentiable: whether to create differentiable hooks
        fused: whether to use fused implementation
    """
    if nesterov and (momentum <= 0 or dampening != 0):
        raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        
    # Initialize momentum buffers if needed
    momentum_buffers = None
    if momentum != 0:
        momentum_buffers = [torch.zeros_like(p) for p in params]
        
    # Convert params and grads to lists if they're not already
    params = list(params)
    grads = [p.grad for p in params]
    
    # Call the Triton implementation
    sgd_step_triton(
        params,
        grads,
        momentum_buffers,
        lr,
        momentum,
        weight_decay,
        dampening,
        nesterov,
        maximize,
    )
