import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n_elements"],
    restore_value=["p_ptr", "exp_avg_ptr"],
)
@triton.jit
def update_fn_kernel(
    p_ptr,
    grad_ptr,
    exp_avg_ptr,
    lr,
    wd,
    beta1,
    beta2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the program ID for the current block
    pid = tl.program_id(axis=0)

    # Determine the starting index for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to handle the case where the block might go out of bounds
    mask = offsets < n_elements

    # Calculate offset pointers for parameters, gradients, and exponential averages
    offset_p_ptr = p_ptr + offsets
    offset_grad_ptr = grad_ptr + offsets
    offset_exp_avg_ptr = exp_avg_ptr + offsets

    # Load data from global memory
    p = tl.load(offset_p_ptr, mask=mask)
    grad = tl.load(offset_grad_ptr, mask=mask)
    exp_avg = tl.load(offset_exp_avg_ptr, mask=mask)

    # Apply step weight decay
    p = p * (1 - lr * wd)

    # Compute the difference between exponential average and gradient
    diff = exp_avg - grad

    # Compute the update value
    update = diff * beta1 + grad

    # Determine the sign of the update and if it can be applied
    can_update = update != 0
    update_sign = tl.where(update > 0, -lr, lr)

    # Update the parameters
    p = p + update_sign * can_update

    # Update the exponential moving average
    exp_avg = diff * beta2 + grad

    # Store the updated parameters and exponential moving average back to global memory
    tl.store(offset_p_ptr, p, mask=mask)
    tl.store(offset_exp_avg_ptr, exp_avg, mask=mask)

def update_fn(
    p: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    lr: float,
    wd: float,
    beta1: float,
    beta2: float,
):
    # Ensure all tensors are on the GPU
    assert all([t.is_cuda for t in (p, grad, exp_avg)])
    n_elements = p.numel()

    # Define the grid size for kernel execution
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch the kernel
    update_fn_kernel[grid](p, grad, exp_avg, lr, wd, beta1, beta2, n_elements)
