import torch
import triton
import triton.language as tl

# Triton kernel for quantizing int8 per row
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "num_stages": 4, "num_warps": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "num_stages": 4, "num_warps": 4}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "num_stages": 4, "num_warps": 4}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "num_stages": 4, "num_warps": 4}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "num_stages": 4, "num_warps": 4}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "num_stages": 4, "num_warps": 4}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "num_stages": 4, "num_warps": 4}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "num_stages": 3, "num_warps": 4}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "num_stages": 4, "num_warps": 2}, num_stages=4, num_warps=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "num_stages": 2, "num_warps": 8}, num_stages=2, num_warps=8),
    ],
    key=["M", "K"],
)
@triton.jit
def quantize_int8_perrow_kernel(
    fpa,  # float pointer to the input matrix
    a,  # int8 pointer to the output matrix
    as_,  # float pointer to the output scales
    M,  # number of rows in the input matrix
    K,  # number of columns in the input matrix
    BLOCK_M: tl.constexpr,  # block size in the rows of the input matrix
    BLOCK_N: tl.constexpr,  # block size in the columns of the input matrix
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    a_ptrs = a + offs_m[:, None] * K + offs_n[None, :]
    fpa_ptrs = fpa + offs_m[:, None] * K + offs_n[None, :]
    N = K

    fpa_block = tl.load(fpa_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)

    abs_fpa_block = tl.abs(fpa_block)
    max_abs_fpa_row = tl.max(abs_fpa_block, axis=1)
    as_ptrs = as_ + offs_m
    tl.store(as_ptrs, max_abs_fpa_row)

    qa_block = tl.libdevice.llrint(127.0 * (fpa_block / max_abs_fpa_row[:, None]))
    tl.store(a_ptrs, qa_block, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# Python wrapper for the Triton kernel
def quantize_int8_perrow(fpa):
    M, K = fpa.shape
    a = torch.empty((M, K), dtype=torch.int8, device=fpa.device)
    as_ = torch.empty((M,), dtype=torch.float16, device=fpa.device)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    quantize_int8_perrow_kernel[grid](fpa, a, as_, M, K)
    return a, as_

# Triton kernel for matrix multiplication
@triton.autotune(
    configs=[
        triton.Config({"SPLIT_K": 1}, num_stages=3, num_warps=8),
        triton.Config({"SPLIT_K": 2}, num_stages=3, num_warps=8),
        triton.Config({"SPLIT_K": 4}, num_stages=3, num_warps=8),
        triton.Config({"SPLIT_K": 8}, num_stages=3, num_warps=8),
        triton.Config({"SPLIT_K": 16}, num_stages=3, num_warps=8),
        triton.Config({"SPLIT_K": 1}, num_stages=4, num_warps=4),
        triton.Config({"SPLIT_K": 2}, num_stages=4, num_warps=4),
        triton.Config({"SPLIT_K": 4}, num_stages=4, num_warps=4),
        triton.Config({"SPLIT_K": 8}, num_stages=4, num_warps=4),
        triton.Config({"SPLIT_K": 16}, num_stages=4, num_warps=4),
        triton.Config({"SPLIT_K": 1}, num_stages=5, num_warps=2),
        triton.Config({"SPLIT_K": 2}, num_stages=5, num_warps=2),
        triton.Config({"SPLIT_K": 4}, num_stages=5, num_warps=2),
        triton.Config({"SPLIT_K": 8}, num_stages=5, num_warps=2),
        triton.Config({"SPLIT_K": 16}, num_stages=5, num_warps=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a,
    b,
    c,
    as_ptr,
    bs_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_
