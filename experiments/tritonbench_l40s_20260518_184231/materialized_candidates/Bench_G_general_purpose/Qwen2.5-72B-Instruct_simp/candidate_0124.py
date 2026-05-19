import triton
import triton.language as tl

@triton.jit
def logsumexp_fwd_kernel(
    x_ptr,  # Pointer to the input tensor
    z_ptr,  # Pointer to the output tensor
    scale,  # Optional scale factor
    D,      # Dimension size of the last dimension
    B,      # Block size
    stride, # Stride of the input tensor
    **meta  # Metadata for the kernel
):
    pid = tl.program_id(axis=0)
    block_start = pid * B
    offsets = block_start + tl.arange(0, B)
    mask = offsets < D
    x = tl.load(x_ptr + offsets * stride, mask=mask, other=-float('inf'))
    
    if scale != 0.0:
        x = x * scale
    
    max_val = tl.max(x, axis=0)
    x = x - max_val
    exp_x = tl.exp(x)
    sum_exp_x = tl.sum(exp_x, axis=0)
    log_sum_exp = tl.log(sum_exp_x) + max_val
    
    tl.store(z_ptr + pid, log_sum_exp)

import torch
import triton
import triton.language as tl

def logsumexp_fwd(x, scale=1.0, dtype=None):
    # Reshape the input tensor to a 2D tensor where the last dimension is D
    B = 1024  # Block size
    D = x.shape[-1]
    x_reshaped = x.view(-1, D)
    N = x_reshaped.shape[0]
    
    # Create an empty output tensor
    z = torch.empty((N,), dtype=x.dtype, device=x.device)
    
    # Determine the appropriate block size
    grid = (N,)
    
    # Launch the kernel
    logsumexp_fwd_kernel[grid](
        x_reshaped,  # Input tensor
        z,           # Output tensor
        scale,       # Scale factor
        D,           # Dimension size
        B,           # Block size
        x.stride(-1) # Stride of the input tensor
    )
    
    # Reshape the result back to the original tensor shape minus the last dimension
    original_shape = x.shape[:-1]
    z = z.view(original_shape)
    
    # Optionally cast the result to the specified data type
    if dtype is not None:
        z = z.to(dtype)
    
    return z
