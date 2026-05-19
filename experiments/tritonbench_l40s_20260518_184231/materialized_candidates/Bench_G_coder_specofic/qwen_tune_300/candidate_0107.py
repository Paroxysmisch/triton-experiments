import torch
import triton
import triton.language as tl
from mamba_ssm.ops.triton.softplus import _softplus_kernel

@triton.jit
def _swiglu_bwd_kernel(
    X, Y, DX, DY, DOUT, OUT, 
    n_elements, 
    BLOCK_N: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_N
    offsets = block_start + tl.arange(0, BLOCK_N)
    mask = offsets < n_elements
    row_idx = offsets // X.shape[1]
    col_idx = offsets % X.shape[1]
    
    x = tl.load(X + row_idx * X.shape[1] + col_idx, mask=mask, other=0.)
    y = tl.load(Y + row_idx * Y.shape[1] + col_idx, mask=mask, other=0.)
    dout = tl.load(DOUT + row_idx * DOUT.shape[1] + col_idx, mask=mask, other=0.)
    
    x_sigmoid = tl.sigmoid(x)
    dx = dout * x_sigmoid * (1. + x * (1. - x_sigmoid))
    tl.store(DX + row_idx * DX.shape[1] + col_idx, dx, mask=mask)
    
    dy = dout * y
    tl.store(DY + row_idx * DY.shape[1] + col_idx, dy, mask=mask)
    
    if RECOMPUTE_OUTPUT:
        out = x * y
        tl.store(OUT + row_idx * OUT.shape[1] + col_idx, out, mask=mask)

def _swiglu_bwd(xy, dout, recompute_output=False):
    if not isinstance(xy, torch.Tensor):
        xy = torch.cat(xy, dim=-1)
    if not xy.is_contiguous():
        xy = xy.contiguous()
    if not dout.is_contiguous():
        dout = dout.contiguous()
    
    batch_shape = xy.shape[:-1]
    xy = xy.reshape(-1, xy.shape[-1])
    dout = dout.reshape(-1, dout.shape[-1])
    dx = torch.empty_like(dout)
    dy = torch.empty_like(dout)
    
    if recompute_output:
        out = torch.empty_like(dout)
    else:
        out = None
    
    n_elements = xy.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_N']),)
    _swiglu_bwd_kernel[grid](
        xy, xy, dx, dy, dout, out, 
        n_elements, 
        RECOMPUTE_OUTPUT=recompute_output
    )
    
    dx = dx.reshape(*batch_shape, dx.shape[-1])
    dy = dy.reshape(*batch_shape, dy.shape[-1])
    
    if recompute_output:
        out = out.reshape(*batch_shape, out.shape[-1])
        return dx, dy, out
    else:
        return dx, dy
