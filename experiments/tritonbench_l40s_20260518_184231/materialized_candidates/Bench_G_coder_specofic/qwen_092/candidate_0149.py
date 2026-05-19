triton
import triton
import triton.language as tl

BLOCK_SIZE = 256

@triton.jit
def update_fn_kernel(
    p_ptr, grad_ptr, exp_avg_ptr, out_ptr, n_elements,
    lr, wd, beta1, beta2, pid=triton.program_id(0)
):
    # Compute the starting offset for this block
    start_idx = pid * BLOCK_SIZE
    # Ensure we only process valid elements
    valid_mask = start_idx < n_elements
    # Load parameters, gradients, and exponential moving averages
    p = tl.load(p_ptr + start_idx, mask=valid_mask)
    grad = tl.load(grad_ptr + start_idx, mask=valid_mask)
    exp_avg = tl.load(exp_avg_ptr + start_idx, mask=valid_mask)
    
    # Apply step weight decay
    p *= (1.0 - lr * wd)
    
    # Calculate the difference between exp_avg and grad
    diff = exp_avg - grad
    
    # Update parameter with momentum term
    p += beta1 * diff
    
    # Sign-based adjustment
    sign = tl.where(p != 0, p / tl.abs(p), 0.0)
    p += sign
    
    # Decay the exponential moving average
    exp_avg *= beta2
    exp_avg += (1.0 - beta2) * grad
    
    # Store the updated parameter and exponential average back to global memory
    tl.store(out_ptr + start_idx, p, mask=valid_mask)
    tl.store(exp_avg_ptr + start_idx, exp_avg, mask=valid_mask)
