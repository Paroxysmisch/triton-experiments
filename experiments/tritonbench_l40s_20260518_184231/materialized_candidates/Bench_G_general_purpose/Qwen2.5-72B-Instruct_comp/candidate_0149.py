import triton
import triton.language as tl

@triton.jit
def update_fn_kernel(
    p_ptr,  # Pointer to the parameters
    grad_ptr,  # Pointer to the gradients
    exp_avg_ptr,  # Pointer to the exponential moving average of past gradients
    n_elements,  # Total number of elements
    lr,  # Learning rate
    wd,  # Weight decay
    beta1,  # Momentum term
    beta2,  # Exponential decay rate for the moving average
    BLOCK_SIZE: tl.constexpr  # Block size
):
    pid = tl.program_id(axis=0)  # Get the program ID
    block_start = pid * BLOCK_SIZE  # Compute the starting offset for the block

    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Compute the offsets for the block
    mask = offsets < n_elements  # Mask to ensure only valid elements are processed

    p = tl.load(p_ptr + offsets, mask=mask)  # Load parameters
    grad = tl.load(grad_ptr + offsets, mask=mask)  # Load gradients
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)  # Load exponential moving average

    # Apply step weight decay
    p = p * (1 - lr * wd)

    # Calculate the difference between the exponential average and the current gradient
    diff = exp_avg - grad

    # Update the parameter with a momentum term
    p = p + beta1 * diff

    # Sign-based adjustment
    sign_diff = tl.where(diff != 0, tl.sign(diff), 0)
    p = p + sign_diff

    # Decay the exponential moving average
    exp_avg = beta2 * exp_avg + (1 - beta2) * grad

    # Store the updated parameter and exponential moving average back to global memory
    tl.store(p_ptr + offsets, p, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)

import torch

def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2, block_size=256):
    # Ensure all tensors are on the same device and are CUDA tensors
    assert p.device.type == 'cuda'
    assert grad.device.type == 'cuda'
    assert exp_avg.device.type == 'cuda'

    # Ensure all tensors have the same shape
    assert p.shape == grad.shape
    assert p.shape == exp_avg.shape

    # Calculate the number of elements
    n_elements = p.numel()

    # Calculate the grid size
    grid = (n_elements + block_size - 1) // block_size

    # Invoke the kernel
    update_fn_kernel[grid, block_size](
        p, grad, exp_avg, n_elements, lr, wd, beta1, beta2, block_size
    )
