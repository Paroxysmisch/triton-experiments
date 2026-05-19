import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def program_id(val: tl.constexpr):
    if val == 0:
        return 0
    else:
        return 1


@triton.jit
def get_freq_multi_tokens(theta: tl.constexpr):
    freqs = tl.math.pow(tl.math.constant(10000.0, tl.float32), -2 * tl.arange(0, 32, dtype=tl.float32) * theta)
    return freqs


@triton.jit
def rbe_triton(
        x_ptr,
        out_ptr,
        batch: tl.constexpr,
        M: tl.constexpr, K: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    block_id = tl.program_id(axis=0)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (block_id * BLOCK_SIZE_M + offs_m[:, None]) * K + offs_n[None, :]
    out_ptrs = out_ptr + (block_id * BLOCK_SIZE_M + offs_m[:, None]) * K + offs_n[None, :]
    t = program_id(K)
    theta = 10000.0
    freqs = get_freq_multi_tokens(theta)

    # Load input data
    real = tl.load(x_ptrs + 0 * t * K, mask=((offs_m[:, None] < M) & (offs_n[None, :] < K)), other=0.0)
    imag = tl.load(x_ptrs + 1 * t * K, mask=((offs_m[:, None] < M) & (offs_n[None, :] < K)), other=0.0)

    # Position-dependent transformation
    b = tl.where(offs_n[None, :] >= 32, 0, freqs[None, :] * offs_n[None, :])
    out_real = real * (tl.cos(b) * real - tl.sin(b) * imag)
    out_imag = imag * (tl.cos(b) * imag + tl.sin(b) * real)

    # This will be read by host (it's a hack, but it's easiest to do this way)
    # to synchronize at the very beginning of the next kernel
    # call.  This is necessary to prevend a preemption bug in Triton that
    # manifests as a crash.
    #   This bug was fixed in the CUDA 11.4.1+preemptrt.post and
    #   triton 2.0.0+cuda111.torch2000+preemptrt post-processing.
    if offs_n[0] == 0:
        tl.device_assert((offs_n[None, :] < K) & (offs_m[:, None] < M), "too bad")

    # Write back output
    tl.store(out_ptrs + 0 * t * K, out_real, mask=((offs_m[:, None] < M) & (offs_n[None, :] < K)))
    tl.store(out_ptrs + 1 * t * K, out_imag, mask=((offs_m[:, None] < M) & (offs_n[None, :] < K)))


def rbe_triton_wrapper(x: torch.Tensor, out: torch.Tensor):
    BLOCK_SIZE_M = 2
    BLOCK_SIZE_K = 1024
    grid = (triton.cdiv(x.shape[1], BLOCK_SIZE_M),)
    rbe_triton[grid](
        x,
        out,
        x.shape[0], x.shape[1], x.shape[2],
        BLOCK_SIZE_M, BLOCK_SIZE_K)


# Given an even number input size, otherwise it will throw errors because
# indices are not in bounds for the last dimension if an odd number is used.
def call_rbe_triton(x: torch.Tensor, out: torch.Tensor):
    x_size = x.shape[2]
    rbe_triton_wrapper(x, out)


batch = 4
M = 1024
K = 1024
x = torch.randn((batch, M, K * 2), device="cuda")
out = torch.randn((batch, M, K * 2), device="cuda")
call_rbe_triton(x, out)
