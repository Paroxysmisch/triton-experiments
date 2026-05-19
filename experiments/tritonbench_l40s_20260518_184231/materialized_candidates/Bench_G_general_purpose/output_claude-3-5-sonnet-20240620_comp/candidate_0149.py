import triton
import triton.language as tl
import torch

@triton.jit
def update_fn_kernel(
    p_ptr,          # pointer to parameters
    grad_ptr,       # pointer to gradients
    exp_avg_ptr,    # pointer to exponential moving averages
    lr,             # learning rate
    wd,            # weight decay
    beta1,         # momentum factor for parameters
    beta2,         # momentum factor for exp avg
    n_elements,    # number of elements
    BLOCK_SIZE: tl.constexpr,  # size of block for parallelization
):
    # Calculate starting offset for this program instance
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset range for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load values from memory
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)
    
    # Apply weight decay
    p = p * (1.0 - lr * wd)
    
    # Calculate update direction using momentum
    diff = exp_avg - grad
    update = tl.where(diff != 0, tl.sign(diff), 0.0)
    
    # Update parameters with momentum
    p = p - lr * (beta1 * update + (1.0 - beta1) * grad)
    
    # Update exponential moving average
    exp_avg = beta2 * exp_avg + (1.0 - beta2) * grad
    
    # Store results back to memory
    tl.store(p_ptr + offsets, p, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)

def update_fn(
    params: torch.Tensor,
    grads: torch.Tensor,
    exp_avg: torch.Tensor,
    lr: float,
    wd: float,
    beta1: float,
    beta2: float
):
    assert params.is_cuda and grads.is_cuda and exp_avg.is_cuda, "All tensors must be CUDA tensors"
    assert params.dtype == grads.dtype == exp_avg.dtype, "All tensors must have the same dtype"
    
    n_elements = params.numel()
    BLOCK_SIZE = 1024  # Can be tuned based on GPU architecture
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    update_fn_kernel[grid](
        params.data_ptr(),
        grads.data_ptr(),
        exp_avg.data_ptr(),
        lr,
        wd,
        beta1,
        beta2,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
