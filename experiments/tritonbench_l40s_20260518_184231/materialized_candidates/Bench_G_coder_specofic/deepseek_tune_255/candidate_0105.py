import math
import torch
import triton
import triton.language as tl
from vllm.model_executor.layers.ops.triton.get_freq import get_freq_multi_tokens

@triton.jit
def rbe_triton(
    x,
    out,
    theta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program id
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    # Compute offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    # Load data from x
    x_mask = offs_m[:, None] < x.shape[1] and offs_k[None, :] < x.shape[2]
    x_real = tl.load(x + offs_m[:, None] * x.strides[1] + offs_k[None, :] * x.strides[2], mask=x_mask, other=0.0).to(tl.float32)
    x_imag = tl.load(x + offs_m[:, None] * x.strides[1] + (offs_k[None, :] + x.shape[2] // 2) * x.strides[2], mask=x_mask, other=0.0).to(tl.float32)
    # Get freqs
    freqs_real, freqs_imag = get_freq_multi_tokens(offs_k, theta=theta)
    # Do rbe
    out_real = x_real * freqs_real - x_imag * freqs_imag
    out_imag = x_real * freqs_imag + x_imag * freqs_real
    # Write back to output
    out_mask = offs_m[:, None] < out.shape[1] and offs_k[None, :] < out.shape[2]
    tl.store(out + offs_m[:, None] * out.strides[1] + offs_k[None, :] * out.strides[2], out_real, mask=out_mask)
    tl.store(out + offs_m[:, None] * out.strides[1] + (offs_k[None, :] + out.shape[2] // 2) * out.strides[2], out_imag, mask=out_mask)

def rbe_triton_wrapper(x, theta=10000.0):
    # Init output tensor
    out = torch.zeros_like(x, dtype=torch.float32, device=x.device)
    # Prepare parameters
    BLOCK_SIZE_M = 2
    BLOCK_SIZE_K = 1024
    num_warps = 8
    # Define grid
    grid = lambda meta: (
        triton.cdiv(x.shape[0], meta["BLOCK_SIZE_M"]),
        triton.cdiv(x.shape[2], meta["BLOCK_SIZE_K"]),
    )
    # Launch kernel
    rbe_triton[grid](x, out, theta, BLOCK_SIZE_M, BLOCK_SIZE_K, num_warps=num_warps)
    return out
