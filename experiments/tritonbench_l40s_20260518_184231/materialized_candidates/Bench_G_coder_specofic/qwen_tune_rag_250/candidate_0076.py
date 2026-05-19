to execute the quantized matrix multiplication. It uses `torch.empty` to allocate tensors for storing intermediate results and final output, and computes grid dimensions for the kernel launch based on input matrix sizes.

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"num_stages": 1, "num_warps": 8}, num_slices=8),
        triton.Config({"num_stages": 2, "num_warps": 8}, num_slices=8),
        triton.Config({"num_stages": 4, "num_warps": 8}, num_slices=8),
        triton.Config({"num_stages": 8, "num_warps": 8}, num_slices=8),
        triton.Config({"num_stages": 1}, num_slices=8),
        triton.Config({"num_stages": 2}, num_slices=8),
        triton.Config({"num_stages": 4}, num_slices=8),
        triton.Config({"num_stages": 8}, num_slices=8),
        triton.Config({"num_stages": 1, "num_warps": 8}, num_slices=8),
        triton.Config({"num_stages": 2, "num_warps": 8}, num_slices=8),
        triton.Config({"num_stages": 4, "num_warps": 8}, num_slices=8),
        triton.Config({"num_stages": 8, "num_warps": 8}, num_slices=8),
        triton.Config({"num_stages": 1}),
        triton.Config({"num_stages": 2}),
        triton.Config({"num_stages": 4}),
        triton.Config({"num_stages": 8}),
        triton.Config({"num_stages": 1, "num_warps": 8}),
        triton.Config({"num_stages": 2, "num_warps": 8}),
        triton.Config({"num_stages": 4, "num_warps": 8}),
        triton.Config({"num_stages": 8, "num_warps": 8}),
    ],
    key=["SPLIT_K", "TK", "TL"],
)
@triton.jit
def quantize_int8_perrow_kernel(
    fpa,
    a,
    as_,
    stride_fpa_n,
    stride_fpa_k,
    stride_a_n,
    stride_a_k,
    stride_as_n,
    stride_as_k,
    M,
    K,
    TK: tl.constexpr,
    TL: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = (M + TK - 1) // TK
    grid_k = (K + TL - 1) // TL
    n = pid
    k = tl.arange(0, TL)
    pa = n * TK + tl.arange(0, TK)
    pb = k
    idx = (pa[:, None] * stride_fpa_n + pb[None, :] * stride_fpa_k)
    mask = (pa < M)[:, None] & (pb < K)[None, :]
    fpa_vals = tl.load(fpa + idx, mask=mask, other=0.0)
    fpa_max = tl.max(tl.abs(fpa_vals), axis=1)
    a_val = tl.cast(tl.math.llrint(127.0 * (fpa_vals / fpa_max[:, None])), tl.int8)
    idx_a = (n * TK + tl.arange(0, TK)) * stride_a_n + k * stride_a_k
    tl.store(a + idx_a, a_val, mask=(n * TK + tl.arange(0, TK) < M)[:, None] & (k < K)[None, :])
    tl.store(
        as_ + n * stride_as_k,
        fpa_max,
        mask=(n < M),
    )

def quantize_int8_perrow(fpa):
    a = torch.empty((fpa.shape[0], fpa.shape[1]), device=fpa.device, dtype=torch.int8)
    as_ = torch.empty((fpa.shape[0]), device=fpa.device, dtype=torch.float16)
    assert fpa.is_contiguous()
    M, K = fpa.shape
    grid_fn = lambda meta: (triton.cdiv(M, meta["TK"]),)
    quantize_int8_perrow_kernel[grid_fn](fpa, a, as_, M, K, K, M, K, 1, 1, M, K)
    return a, as_

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "SPLIT_K": 1}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "SPLIT_K": 1}, num_stages=5, num_warps=2
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "SPLIT_K": 1}, num_stages=5, num_warps=2
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "SPLIT_K": 1}, num_stages=4, num_warps=2
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "SPLIT_K": 1}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "SPLIT_K": 1},
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "SPLIT_K": 1},
            num_stages=2,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "SPLIT_K
