import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N", "HAS_RESIDUAL", "STORE_RESIDUAL_OUT", "IS_RMS_NORM", "HAS_BIAS"],
)
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    Y,  # pointer to the output
    A,  # pointer to the input
    RESIDUAL,  # pointer to the residual
    RESIDUAL_OUT,  # pointer to the residual
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_ck,
    stride_resm, stride_resk,
    stride_resoutm, stride_resoutk,
    M,  # number of rows in A
    N,  # number of columns in A
    eps,  # epsilon to avoid division by zero
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    RESIDUAL_DTYPE: tl.constexpr,
):
    row = tl.program_id(0)
    Y += row * stride_am
    A += row * stride_am
    W += row * stride_bk
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_resm
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_resoutm
    if HAS_BIAS:
        B += row * stride_bn
    cols = tl.arange(0, BLOCK_N)
    a = tl.load(A + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0).to(tl.float32)
        a += residual
    if STORE_RESIDUAL_OUT:
        tl.store(RESIDUAL_OUT + cols, a, mask=cols < N)
    if IS_RMS_NORM:
        a_rms = tl.where(cols < N, a * a, 0.0)
        mean = tl.sum(a_rms, axis=0) / N
    else:
        mean = tl.sum(a, axis=0) / N
    a -= mean
    if HAS_BIAS:
        b = tl.load(B + cols, mask=cols < N).to(tl.float32)
    else:
        b = tl.zeros([BLOCK_N], dtype=tl.float32)
    mean = tl.where(cols < N, mean, 0.0)
    tl.store(Mean + row, mean, mask=row < M)
    var = tl.sum(a * a, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd, mask=row < M)
    mask = cols < N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    a = a * rstd * w + b
    if OUTPUT_DTYPE == tl.float16:
        a = a.to(tl.float16)
    tl.store(Y + cols, a, mask=mask)

def _layer_norm_fwd(
    a, weight, bias, eps, residual=None, out_dtype=None, residual_dtype=None, is_rms_norm=False
):
    a_arg = a.clone() if RESIDUAL_APPROXIMATION == "additive" else a
    M, N = a.shape
    assert a.shape == weight.shape and a.shape == bias.shape
    a = a.view((M, -1))
    a_stride_m, a_stride_n = a.stride()
    weight_stride_m, weight_stride_n = weight.stride()
    bias_stride_m, bias_stride_n = bias.stride()
    assert weight_stride_m == 0 and bias_stride_m == 0
    rstd = torch.empty((M,), dtype=a.dtype, device=a.device)
    mean = torch.empty((M,), dtype=torch.float32, device=a.device)
    B, N = weight.shape
    y = torch.empty_like(a)
    if residual is not None:
        assert a.shape == residual.shape
        residual_stride_m, residual_stride_n = residual.stride()
        residual_out = torch.empty_like(residual)
        _layer_norm_fwd_1pass_kernel[(M,)](
            y,
            a,
            residual,
            residual_out,
            weight,
            bias,
            mean,
            rstd,
            a_stride_m,
            a_stride_n,
            weight_stride_m,
            weight_stride_n,
            residual_stride_m,
            residual_stride_n,
            M,
            N,
            eps,
            is_rms_norm,
            BLOCK_N=N,
            HAS_RESIDUAL=True,
            STORE_RESIDUAL_OUT=True,
            HAS_BIAS=True,
            OUTPUT_DTYPE=getattr(torch, out_dtype) if out_dtype is not None else None,
            RESIDUAL_DTYPE=getattr(torch, residual_dtype) if residual_dtype is not None else None,
        )
        return y, mean, rstd, residual_out
    else:
        _layer_norm_fwd_1pass_kernel[(M,)](
            y,
            a,
            weight,
            bias,
            mean,
            rstd,
            a_stride_m,
            a_stride_n,
            weight_stride_m,
            weight_stride_n,
            M,
            N,
            eps,
            is_rms_norm,
            BLOCK_N=N,
            HAS_RESIDUAL=False,
            STORE_RESIDUAL_OUT=False,
            HAS_BIAS=True,
            OUTPUT_DTYPE=getattr(torch, out_dtype) if out_dtype is not None else None,
        )
        return y, mean, rstd

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32}, num_stages=1, num_warps=8),
        triton.Config({"BLOCK_N": 32}, num_stages=1, num_warps=16),
        triton.Config({"BLOCK_N": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_N": 32}, num_stages=2, num_warps=16),
