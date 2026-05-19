import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_mul_activation_kernel(
    x_ptr, 
    bias_ptr, 
    in_ptr, 
    out_ptr, 
    scale, 
    n_elements, 
    activation_type, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first")
    b = tl.load(bias_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first")
    i = tl.load(in_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first")

    out = x + b + scale * i

    if activation_type == 1:
        out = tl.maximum(out, 0.0)
    elif activation_type == 2:
        out = 1.0 / (1.0 + tl.exp(-out))

    tl.store(out_ptr + offsets, out, mask=mask, eviction_policy="evict_last")

def fused_add_mul_activation_torch(x, in_t, bias, scale=1.0, activation='sigmoid'):
    assert x.is_cuda, "x must be a CUDA tensor"
    assert in_t.is_cuda, "in_t must be a CUDA tensor"
    assert bias.is_cuda, "bias must be a CUDA tensor"
    activation_map = {
        'none': 0,
        'relu': 1,
        'sigmoid': 2
    }
    activation_type = activation_map.get(activation, 2)
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_add_mul_activation_kernel[grid](
        x, 
        bias, 
        in_t, 
        out, 
        scale, 
        n_elements, 
        activation_type, 
        BLOCK_SIZE=1024
    )
    return out
