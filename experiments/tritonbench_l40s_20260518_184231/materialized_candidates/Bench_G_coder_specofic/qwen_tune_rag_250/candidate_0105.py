import triton
import triton.language as tl
import torch
from torch import Tensor
from typing import Optional
from xopes.utils import get_freq_multi_tokens

# rbe triton kernel
@triton.jit
def rbe_triton(
    x,
    out,
    freq_real,
    freq_imag,
    stride_xm,
    stride_xk,
    stride_zn,
    stride_zk,
    stride_freq_real_m,
    stride_freq_real_k,
    stride_freq_imag_m,
    stride_freq_imag_k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    The input is of shape [batch, M, K]
    The output is of shape [batch, M, K]
    """
    program_id = tl.program_id(axis=0)
    X = x + program_id * stride_xm
    Out = out + program_id * stride_zn
    freq_real = freq_real + program_id * stride_freq_real_m
    freq_imag = freq_imag + program_id * stride_freq_imag_m

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    real = tl.load(X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    imag = tl.load(X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk + 1)

    pi = 3.141592653589793
    real = real * tl.cos(freq_real * pi * offs_k[None, :] * freq_real)
    imag = imag * tl.sin(freq_imag * pi * offs_k[None, :] * freq_imag)
    out_real = tl.load(freq_real)
    out_imag = tl.load(freq_imag)
    tl.store(Out + offs_m[:, None] * 1 + offs_k[None, :] * stride_zk, out_real)
    tl.store(Out + offs_m[:, None] * 1 + offs_k[None, :] * stride_zk + 1, out_imag)

# rbe triton wrapper
def rbe_triton_wrapper(x: Tensor, device: Optional[torch.device] = None):
    batch, M, K = x.shape
    out = torch.empty(batch, M, K, device=x.device, dtype=x.dtype)
    freq_real, freq_imag = get_freq_multi_tokens(device=device, max_position_embeddings=M, dtype=torch.float32)
    grid = lambda META: (batch, )
    rbe_triton[grid](
        x, out,
        freq_real, freq_imag,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        freq_real.stride(0), freq_real.stride(1),
        freq_imag.stride(0), freq_imag.stride(1),
        BLOCK_SIZE_M=2,
        BLOCK_SIZE_K=1024,
        num_warps=8,
    )
    return out
