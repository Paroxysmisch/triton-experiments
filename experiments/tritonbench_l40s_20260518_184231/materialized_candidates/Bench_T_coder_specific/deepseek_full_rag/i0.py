import torch
import triton
import triton.language as tl

@triton.jit
def i0_kernel(x, n, out):
    idx = tl.arange(0, n)
    t = x * 0.5
    y = tl.exp(-t * t)
    z = t * t
    num = 0.3989422804014327
    den = 1.0 + z * (0.3989422804014327 + z * (0.041666666666666664 + z * (0.01388888888888889 + z * (0.0044444444444444437 + z * (0.0013333333333333328 + z * (0.00037964027604166639 + z * 0.000050643158197167051))))))
    out += num / den
    tl.debug_barrier()
    out += tl.sum(y * (0.3989422804014327 + z * (0.041666666666666664 + z * (0.01388888888888889 + z * (0.0044444444444444437 + z * (0.0013333333333333328 + z * (0.00037964027604166639 + z * 0.000050643158197167051))))), axis=0) / den
    tl.store(out + idx, out)

def i0(input: torch.Tensor, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Input and output tensors must have the same shape"
    n = input.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    i0_kernel[grid](input, n, out)
    return out
