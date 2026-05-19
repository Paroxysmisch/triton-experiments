import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def linear_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias_ptrs = bias_ptr + offs_n
        bias = tl.load(bias_ptrs, mask=(offs_n < N), other=0.0)
        accumulator += bias[None, :]

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.jit
def logsumexp_fwd_kernel(
    x_ptr,
    z_ptr,
    scale,
    D: tl.constexpr,
    B: tl.constexpr,
    HAS_SCALE: tl.constexpr
):
    i_n, i_d = tl.program_id(0), tl.program_id(1)
    o_d = i_d * B + tl.arange(0, B)
    mask = o_d < D

    x = tl.load(x_ptr + i_n * D + o_d, mask=mask, other=-float('inf'))
    if HAS_SCALE:
        x = x * scale
    m = tl.max(x, axis=0)
    lse = tl.log(tl.sum(tl.exp(x - m), axis=0)) + m
    tl.store(z_ptr + i_n * tl.cdiv(D, B) + i_d, lse)

def logsumexp_fwd(
    x: torch.Tensor,
    scale: Optional[float] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    shape = x.shape
    x = x.view(-1, shape[-1])
    N, D = x.shape
    B = min(triton.next_power_of_2(D), 64 * 1024)
    ND = triton.cdiv(D, B)

    z = torch.empty((N, ND), device=x.device, dtype=torch.float32)
    HAS_SCALE = scale is not None
    logsumexp_fwd_kernel[(N, ND)](
        x_ptr=x,
        z_ptr=z,
        scale=scale,
        D=D,
        B=B,
        HAS_SCALE=HAS_SCALE
    )
    z = z.logsumexp(-1)
    if dtype is not None:
        z = z.to(dtype)
    return z.view(*shape[:-1])

@triton.jit
def subtract_lse_kernel(
    x_ptr,
    lse_ptr,
    y_ptr,
    D: tl.constexpr,
    B: tl.constexpr,
):
    i_n, i_d = tl.program_id(0), tl.program_id(1)
    o_d = i_d * B + tl.arange(0, B)
    mask = o_d < D

    x = tl.load(x_ptr + i_n * D + o_d, mask=mask, other=0.0)
    lse = tl.load(lse_ptr + i_n)
    y = x - lse
    tl.store(y_ptr + i_n * D + o_d, y, mask=mask)

def subtract_lse(x: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    N, D = x.shape
    B = min(triton.next_power_of_2(D), 64 * 1024)
    ND = triton.cdiv(D, B)
    y = torch.empty_like(x)
    subtract_lse_kernel[(N, ND)](
        x_ptr=x,
        lse_ptr=lse,
        y_ptr=y,
        D=D,
        B=B
    )
    return y

def log_softmax_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dim: int = -1,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    if dtype is not None:
        input = input.to(dtype)
    input_3d = input.view(-1, input.size(-1))
    M, K = input_3d.size()
    N, _ = weight.size()

    linear_output = torch.empty((M, N), device=input.device, dtype=torch.float32)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    HAS_BIAS = bias is not None
    bias_ptr = bias.data_ptr() if HAS_BIAS else 0

    linear_kernel[grid](
        input_3d.data_ptr(),
        weight.data_ptr(),
        bias_ptr,
        linear_output.data_ptr(),
        M, N, K,
        input_3d.stride(0), input_3d.stride(1),
        weight.stride(0), weight.stride(1),
        linear_output.stride(0), linear_output.stride(1),
        HAS_BIAS=HAS_BIAS,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    new_shape = input.shape[:-1] + (N,)
    linear_output = linear_output.view(new_shape)
    original_ndim = linear_output.dim()
    dim = dim if dim >= 0 else original_ndim + dim

    if dim != original_ndim - 1:
        perm = list(range(original_ndim))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        linear_output = linear_output.permute(perm)
        permuted_shape = linear_output.shape
        linear_output = linear_output.contiguous().view(-1, permuted_shape[-1])

    lse = logsumexp_fwd(linear_output, dtype=torch.float32)
    log_softmax_output = subtract_lse(linear_output, lse)

    if dim != original_ndim - 1:
        log_softmax_output = log_softmax_output.view(permuted_shape).permute(perm).contiguous()
        log_softmax_output = log_softmax_output.view(new_shape)

    return log_softmax_output
