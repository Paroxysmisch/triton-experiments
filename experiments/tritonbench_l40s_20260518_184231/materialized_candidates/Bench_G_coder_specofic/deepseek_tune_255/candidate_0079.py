import torch
import triton
import triton.language as tl

# Triton kernel for quantize and transpose
@triton.jit
def _quantize_global_transpose(
    A,
    B,
    M,
    N,
    absmax_inv,
    stride_am,
    stride_an,
    stride_bk,
    stride_bkn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // N
    pid_n = pid % N
    group_id = pid_m % GROUP_M
    # re-order the pid so that we cover all the blocks
    pid_m = GROUP_M * (pid_m // GROUP_M) + (pid_m + 1) % GROUP_M

    offs_am = (
        (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) * stride_am
        + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) * stride_an
    )
    offs_bn = (
        (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) * stride_bk
        + group_id * stride_bkn
        + (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    )
    # load data from A
    a_vals = tl.load(A + offs_am, mask=(offs_am < M * stride_am) & (offs_am < N * stride_an), other=0.0)
    # apply quantization
    q_vals = a_vals * absmax_inv
    # store the result in B
    tl.store(B + offs_bn, q_vals, mask=(offs_bn < N * stride_bk) & (offs_bn < GROUP_M * stride_bkn))

# Wrapper function for launching Triton kernel
def quantize_global_transpose(A: torch.Tensor, absmax: float, B: torch.Tensor):
    M, N = A.shape
    BLOCK_M, BLOCK_N = 32, 32
    GROUP_M = 4
    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert A.is_cuda and B.is_cuda
    absmax_inv = 127.0 / absmax
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    _quantize_global_transpose[grid](
        A,
        B,
        M,
        N,
        absmax_inv,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
    )

# Function to call Triton kernel
def call_quantize_global_transpose(A: torch.Tensor, B: torch.Tensor):
    absmax = float(A.abs().max())
    quantize_global_transpose(A, absmax, B)
