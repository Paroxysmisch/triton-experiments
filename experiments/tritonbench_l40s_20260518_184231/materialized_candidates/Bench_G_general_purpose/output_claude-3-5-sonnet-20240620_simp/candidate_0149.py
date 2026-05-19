import triton
import triton.language as tl
import torch

@triton.jit
def update_fn_kernel(
    p_ptr,          # pointer to parameters
    grad_ptr,       # pointer to gradients
    exp_avg_ptr,    # pointer to exp_avg (momentum)
    n_elements,     # number of elements
    learning_rate,  # learning rate
    beta1,         # momentum factor
    weight_decay,  # weight decay factor
    BLOCK_SIZE: tl.constexpr,  # size of parallel blocks
):
    # Calculate the program ID and check if it's within bounds
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create a mask for the current block
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load data from global memory
    p = tl.load(p_ptr + offs, mask=mask)
    grad = tl.load(grad_ptr + offs, mask=mask)
    exp_avg = tl.load(exp_avg_ptr + offs, mask=mask)
    
    # Update exponential moving average (momentum)
    exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    
    # Calculate update with weight decay
    update = exp_avg + weight_decay * p
    
    # Update parameters
    p = p - learning_rate * update
    
    # Store results back to global memory
    tl.store(p_ptr + offs, p, mask=mask)
    tl.store(exp_avg_ptr + offs, exp_avg, mask=mask)

def update_fn(
    p: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    learning_rate: float = 0.01,
    beta1: float = 0.9,
    weight_decay: float = 0.0,
):
    assert p.is_cuda and grad.is_cuda and exp_avg.is_cuda
    assert p.is_contiguous() and grad.is_contiguous() and exp_avg.is_contiguous()
    n_elements = p.numel()
    
    # Define block size and compute grid
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    update_fn_kernel[grid](
        p.data_ptr(),
        grad.data_ptr(),
        exp_avg.data_ptr(),
        n_elements,
        learning_rate,
        beta1,
        weight_decay,
        BLOCK_SIZE=BLOCK_SIZE,
    )
