import torch
import triton
import triton.language as tl
from torch import Tensor
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.triton_heuristics import grid
from torch._inductor import triton_helpers

@triton.jit
def batch_norm_fwd_fused(
    X, Y, W, B, Mean, Rstd, stride, N, HW, C, GROUPS, M, eps, IS_RMS_NORM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, IS_TRT: tl.constexpr
):
    # Compute the program's ID and the offset for the current element
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Load input data and weights
    mask = offset < N * HW
    x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N / HW
        tl.store(Mean + pid, mean)
        xbar = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N / HW
    else:
        xbar = tl.where(mask, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N / HW
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + pid, rstd)
    # Compute the output
    mask = offset < N * C
    w = tl.load(W + offset, mask=mask).to(tl.float32)
    if bias is not None:
        b = tl.load(B + offset, mask=mask).to(tl.float32)
    # Normalize and apply linear transformation
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if bias is not None else x_hat * w
    # Write output
    if IS_TRT:
        offs_n = offset // C
        offs_d = offset % C
        y = tl.reshape(y, (M, N, C))
        y = tl.transpose(y, (1, 0, 2))
        y = tl.reshape(y, (N * M, C))
        mask = offs_n < N
        tl.store(Y + offset, y, mask=mask)
    else:
        tl.store(Y + offset, y, mask=mask)

def batch_norm(
    x: Tensor, 
    weight: Tensor, 
    bias: Tensor, 
    running_mean: Tensor, 
    running_var: Tensor, 
    training: bool = False, 
    momentum: float = 0.1, 
    eps: float = 1e-05, 
    cudnn_benchmark: bool = False
) -> Tensor:
    # Get strides and shape information
    stride = x.stride(0)
    N, HW, C = x.shape
    x_ = x.view(N * HW, C)
    M = C
    GROUPS = C
    # Determine if using RMSNorm
    is_rms_norm = False
    if weight is None:
        is_rms_norm = True
    # Create output tensor
    y = torch.empty_like(x)
    # Create grid and stream for Triton kernel
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']),)
    with torch.cuda._DeviceGuard(0):
        batch_norm_fwd_fused[grid](
            x_, y, weight, bias, running_mean, running_var, 
            stride, N, HW, C, GROUPS, M, eps, is_rms_norm, 
            GROUPS, IS_TRT=False
        )
    return y
