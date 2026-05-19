import torch
import triton
import triton.language as tl
from triton.language.libdevice import hardsigmoid as _hardsigmoid

@triton.jit
def _batch_norm_fwd(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    momentum,
    eps,
    N,
    H,
    C,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    IS_HARDSIGMOID: tl.constexpr,
):
    # compute indices
    row_start = tl.program_id(0) * GROUP_SIZE_M
    row_step = tl.num_programs(0) * GROUP_SIZE_M
    for row_idx in tl.range(row_start, N * H, row_step):
        cols = tl.arange(0, BLOCK_SIZE_C)
        idx = row_idx + cols
        mask = idx < N * H
        # compute mean and variance
        x = tl.load(X + idx, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x) / (N * H)
        x_zm = tl.where(mask, x - mean, 0.0)
        var = tl.sum(x_zm * x_zm) / (N * H)
        rstd = 1 / tl.sqrt(var + eps)
        # update running estimate
        mean_old = tl.load(Mean + tl.program_id(0))
        rstd_old = tl.load(Rstd + tl.program_id(0))
        mean_new = mean_old * momentum + mean * (1 - momentum)
        rstd_new = rstd_old * momentum + rstd * (1 - momentum)
        tl.store(Mean + tl.program_id(0), mean_new)
        tl.store(Rstd + tl.program_id(0), rstd_new)
        # update output
        c = tl.program_id(1)
        mask = cols + c * N * H < N * H
        x_hat = (x_zm * rstd).to(tl.float16)
        w = tl.load(W + c, mask=mask, other=0.0).to(tl.float16)
        y = x_hat * w
        if B is not None:
            b = tl.load(B + c, mask=mask, other=0.0).to(tl.float16)
            y = y + b
        if IS_HARDSIGMOID:
            y = _hardsigmoid(y)
        # write-back
        tl.store(Y + idx, y, mask=mask)

def fused_hardsigmoid_batch_norm(x, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5, inplace=False):
    # check constraints
    assert x.is_contiguous()
    assert running_mean.is_contiguous()
    assert running_var.is_contiguous()
    assert running_mean.shape[0] == x.shape[1]
    if weight is not None:
        assert weight.is_contiguous()
        assert weight.shape[0] == x.shape[1]
    if bias is not None:
        assert bias.is_contiguous()
        assert bias.shape[0] == x.shape[1]
    # set up meta-parameters
    N, H, C = x.shape
    x = x.view(N * H, C)
    M, C = x.shape
    # allocate output
    y = torch.empty_like(x)
    # enqueue kernel
    def grid(meta):
        return (triton.cdiv(M, meta["GROUP_SIZE_M"]), C)
    _batch_norm_fwd[grid](
        x, y, weight, bias,
        running_mean, running_var,
        momentum, eps,
        N, H, C,
        IS_HARDSIGMOID=True,
        inplace=inplace
    )
    return y
