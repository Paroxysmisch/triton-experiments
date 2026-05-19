import torch
import triton
import triton.language as tl

@triton.jit
def logit_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    eps_val: tl.constexpr,
    has_eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask)

    # Clamp if has_eps is True (1), else pass through
    if has_eps:
        z = tl.minimum(tl.maximum(x, eps_val), 1.0 - eps_val)
    else:
        z = x

    # Compute log(z / (1 - z))
    one = tl.full(z.shape, 1.0, z.dtype)
    denominator = one - z
    ratio = z / denominator
    y = tl.log(ratio)

    # Store result
    tl.store(output_ptr + offsets, y, mask=mask)

def logit(input, eps=None, *, out=None):
    # Ensure input is contiguous
    input = input.contiguous()
    
    if out is None:
        out = torch.empty_like(input)
    else:
        out = out.contiguous()
    
    n_elements = input.numel()
    
    # Kernel configuration
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Prepare kernel arguments
    has_eps = 1 if eps is not None else 0
    eps_val = eps if eps is not None else 0.0  # Unused when has_eps is 0
    
    # Launch kernel
    logit_kernel[grid](
        input.data_ptr(),
        out.data_ptr(),
        n_elements,
        eps_val,
        has_eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
