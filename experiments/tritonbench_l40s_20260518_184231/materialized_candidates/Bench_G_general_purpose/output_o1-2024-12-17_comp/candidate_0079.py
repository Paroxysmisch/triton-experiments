import triton
import triton.language as tl
import torch

@triton.jit
def _quantize_global_transpose(
    A_ptr, B_ptr,
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    absmax_inv,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
    group_size = GROUP_M * num_pid_n
    group_id = pid // group_size
    within_group_id = pid % group_size
    pid_m = group_id * GROUP_M + within_group_id // num_pid_n
    pid_n = within_group_id % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    A_offs = offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    B_offs = offs_n[:, None] * stride_bm + offs_m[None, :] * stride_bn

    a = tl.load(A_ptr + A_offs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    q = a * absmax_inv
    q_clamped = tl.max(tl.min(q, 127.0), -128.0)
    q_int8 = q_clamped.to(tl.int8)
    tl.store(B_ptr + B_offs, q_int8, mask=mask_m[None, :] & mask_n[:, None])

def quantize_global_transpose(A, BLOCK_M=128, BLOCK_N=128, GROUP_M=4):
    assert A.is_cuda, "Input must be a CUDA tensor."
    M, N = A.shape
    absmax = A.abs().max()
    absmax_inv = (127.0 / absmax) if absmax != 0 else 0.0

    B = torch.empty((N, M), device=A.device, dtype=torch.int8)

    grid = lambda meta: (
        ((M + BLOCK_M - 1) // BLOCK_M + GROUP_M - 1) // GROUP_M
    ) * ((N + BLOCK_N - 1) // BLOCK_N) * GROUP_M

    _quantize_global_transpose[grid](
        A, B,
        M, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        absmax_inv,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUP_M=GROUP_M
    )
    return B
