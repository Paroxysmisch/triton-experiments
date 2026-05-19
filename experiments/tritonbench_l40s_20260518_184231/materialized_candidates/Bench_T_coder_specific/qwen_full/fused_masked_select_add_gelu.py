import torch
import triton
import triton.language as tl
from triton.language.libdevice import gelu, gelu_approx

@triton.jit
def fused_masked_add_gelu_kernel(X, M, O, A, OUT, stride_x, stride_y, stride_z, stride_m, stride_o, stride_out_m,
                                 stride_out_n, n_elements, alpha, approximate, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Offsetted pointers
    x = tl.load(X + offsets * stride_x, mask=mask)
    m = tl.load(M + offsets * stride_m, mask=mask)
    o = tl.load(O + offsets * stride_o, mask=mask)
    # Fused operations
    s = tl.where(m, x + alpha * o, 0.)
    if approximate == 'tanh':
        y = gelu_approx(s)
    else:
        y = gelu(s)
    # Store output
    tl.store(OUT + offsets * stride_out_m, y, mask=mask)

def fused_masked_add_gelu(x, m, o, *, alpha=1, approximate='none', out=None):
    # Reshape input for processing
    x_ = x.unsqueeze(0) if x.ndim == 1 else x
    m_ = m.unsqueeze(0) if m.ndim == 1 else m
    o_ = o.unsqueeze(0) if o.ndim == 1 else o
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(x_, dtype=x_.dtype)
    else:
        assert out.shape == x.shape
    out_ = out.unsqueeze(0) if out.ndim == 1 else out
    # Grid definition for Triton kernel launch
    grid = lambda meta: (triton.cdiv(x_.numel(), meta['BLOCK_SIZE']),)
    # Launch Triton kernel
    fused_masked_add_gelu_kernel[grid](x_, m_, o_, alpha, approximate, BLOCK_SIZE=1024,
                                       stride_x=x_.stride(0), stride_y=m_.stride(0), stride_z=o_.stride(0),
                                       stride_m=x_.shape[0], stride_o=m_.shape[0], stride_out_m=out_.shape[0],
                                       stride_out_n=out_.shape[1], n_elements=x_.numel())
    return out_.squeeze(0) if out_.ndim > x.ndim else out_
