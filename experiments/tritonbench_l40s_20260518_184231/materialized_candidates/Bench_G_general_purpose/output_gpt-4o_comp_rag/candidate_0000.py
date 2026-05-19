import triton
import triton.language as tl
import torch

@triton.jit
def _swiglu_fwd_kernel(
    X_ptr, Y_ptr, OUT_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    
    block_m = pid // num_blocks_n
    block_n = pid % num_blocks_n
    
    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    X = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    Y = tl.load(Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    sigmoid_X = 1 / (1 + tl.exp(-X))
    result = X * sigmoid_X * Y
    
    tl.store(OUT_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn, result, mask=mask_m[:, None] & mask_n[None, :])

def _swiglu_fwd(xy: torch.Tensor, ncols: int):
    x, y = torch.split(xy, ncols, dim=1)
    x = x.contiguous()
    y = y.contiguous()
    
    M, N = x.shape
    out = torch.empty_like(x)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    _swiglu_fwd_kernel[grid](
        x, y, out,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=128, BLOCK_N=128
    )
    
    return out
