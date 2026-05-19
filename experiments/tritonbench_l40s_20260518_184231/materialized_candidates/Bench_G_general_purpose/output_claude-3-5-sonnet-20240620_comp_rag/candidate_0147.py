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
    p_ptr,           # Pointer to parameters
    grad_ptr,        # Pointer to gradients
    exp_avg_ptr,     # Pointer to exponential moving averages
    lr,              # Learning rate
    wd,              # Weight decay
    beta1,           # First momentum factor
    beta2,           # Second momentum factor
    n_elements,      # Total number of elements to process
    BLOCK_SIZE: tl.constexpr,  # Size of each block (auto-tuned)
):
    # Get the program ID for this instance
    pid = tl.program_id(axis=0)
    
    # Calculate starting offset for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Calculate offset pointers
    offset_p_ptr = p_ptr + offsets
    offset_grad_ptr = grad_ptr + offsets
    offset_exp_avg_ptr = exp_avg_ptr + offsets
    
    # Load values from memory
    p = tl.load(offset_p_ptr, mask=mask)
    grad = tl.load(offset_grad_ptr, mask=mask)
    exp_avg = tl.load(offset_exp_avg_ptr, mask=mask)
    
    # Apply stepweight decay
    p = p * (1 - lr * wd)
    
    # Calculate difference between momentum and gradient
    diff = exp_avg - grad
    
    # Calculate update with momentum
    update = diff * beta1 + grad
    
    # Apply sign-based update
    can_update = update != 0
    update_sign = tl.where(update > 0, -lr, lr)
    p = p + update_sign * can_update
    
    # Update exponential moving average
    exp_avg = diff * beta2 + grad
    
    # Store results back to memory
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
    # Verify inputs are CUDA tensors
    assert all([t.is_cuda for t in (p, grad, exp_avg)])
    n_elements = p.numel()
    
    # Define grid size based on number of elements
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    # Launch kernel
    update_fn_kernel[grid](
        p, grad, exp_avg,
        lr, wd, beta1, beta2,
        n_elements
    )
