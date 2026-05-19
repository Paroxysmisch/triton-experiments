import torch
import triton
import triton.language as tl


@triton.jit
def logit_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input + offsets, mask=mask)
    
    eps = tl.full([BLOCK_SIZE], fill_value=1e-6, dtype=tl.float32)
    eps = tl.broadcast_to(eps, x.shape)
    
    x_clipped = tl.clip(x, eps, 1 - eps)
    y = tl.log(x_clipped / (1 - x_clipped))
    tl.store(output + offsets, y, mask=mask)


def logit(input, eps=None, *, out=None):
    assert input.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.bool)
    assert eps is None or isinstance(eps, float)
    if eps is not None:
        assert eps > 0.0
        assert eps < 0.5
    if out is not None:
        assert out.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.bool)
        assert out.shape == input.shape
    else:
        out = torch.empty_like(input, dtype=input.dtype)
    
    n_dims = input.ndim
    if n_dims == 0:
        raise RuntimeError("logit only accepts inputs of 1 or more dimensions")
    elif n_dims == 1:
        n_elements = out.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        logit_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    else:
        out = out.contiguous()
        input = input.contiguous()
        n_flat_dims = torch.prod(torch.tensor(input.shape[:-1])).item()
        n_elements = out.numel()
        grid = lambda meta: (n_flat_dims,)
        logit_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)
    
    return out
