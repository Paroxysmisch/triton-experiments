import triton
import triton.language as tl

@triton.jit
def update_fn_kernel(
    p_ptr, grad_ptr, exp_avg_ptr,
    lr, momentum, weight_decay,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Compute the block index
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements

    # Load parameters, gradients, and exponential moving averages
    p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0)

    # Update exponential moving average
    exp_avg = momentum * exp_avg + (1 - momentum) * grad

    # Apply weight decay
    p = p - lr * (exp_avg + weight_decay * p)

    # Store updated parameters and averages back to global memory
    tl.store(p_ptr + offsets, p, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)

import torch

def update_fn(p, grad, exp_avg, lr, momentum, weight_decay):
    # Ensure inputs are CUDA tensors
    assert p.is_cuda and grad.is_cuda and exp_avg.is_cuda

    # Number of elements
    n_elements = p.numel()

    # Define block size
    BLOCK_SIZE = 1024  # You can adjust this based on your GPU's capability

    # Calculate grid size
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    update_fn_kernel[grid_size](
        p, grad, exp_avg,
        lr, momentum, weight_decay,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
