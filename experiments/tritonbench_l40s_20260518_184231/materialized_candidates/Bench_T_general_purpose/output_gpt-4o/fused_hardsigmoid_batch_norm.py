import triton
import triton.language as tl

@triton.jit
def fused_hardsigmoid_batch_norm_kernel(
    x_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    out_ptr, C, eps, inplace, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < C

    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask)
    running_mean = tl.load(running_mean_ptr + offsets, mask=mask)
    running_var = tl.load(running_var_ptr + offsets, mask=mask)
    weight = tl.load(weight_ptr + offsets, mask=mask) if weight_ptr else 1.0
    bias = tl.load(bias_ptr + offsets, mask=mask) if bias_ptr else 0.0

    # Batch normalization
    normed = (x - running_mean) / tl.sqrt(running_var + eps)
    normed = normed * weight + bias

    # Hardsigmoid activation
    if inplace:
        x = normed
    else:
        x = normed

    hardsigmoid = tl.maximum(0, tl.minimum(1, x * 0.2 + 0.5))

    # Store result
    tl.store(out_ptr + offsets, hardsigmoid, mask=mask)

import torch

def fused_hardsigmoid_batch_norm(x, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5, inplace=False):
    # Check dimensions
    assert x.dim() >= 2, "Input tensor must have at least 2 dimensions"
    C = x.shape[1]  # Assuming NCHW format

    # Allocate output tensor
    out = x if inplace else torch.empty_like(x)

    # Launch Triton kernel
    BLOCK_SIZE = 1024  # Choose a suitable block size
    grid = lambda meta: (triton.cdiv(C, meta['BLOCK_SIZE']),)
    
    fused_hardsigmoid_batch_norm_kernel[grid](
        x, running_mean, running_var, weight, bias, out,
        C, eps, inplace, BLOCK_SIZE=BLOCK_SIZE
    )

    return out
