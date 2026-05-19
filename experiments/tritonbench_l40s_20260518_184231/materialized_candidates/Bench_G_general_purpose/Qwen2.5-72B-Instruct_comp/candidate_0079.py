import triton
import triton.language as tl

@triton.jit
def _quantize_global_transpose(
    A, B, absmax_inv, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr,
    stride_am, stride_an, stride_bm, stride_bn
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    pid_m = first_pid_m + (pid % GROUP_M)
    pid_n = (pid % num_pid_in_group) // GROUP_M

    # bounds checking
    if pid_m >= num_pid_m:
        return

    # initialize pointers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    A_ptr = A + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
    B_ptr = B + (offs_n[:, None] * stride_bm + offs_m[None, :] * stride_bn)

    # load data
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    a = tl.load(A_ptr, mask=mask, other=0.0)

    # quantize
    a_quantized = tl.cast(tl.where(mask, a * absmax_inv, 0.0), tl.int8)

    # store data
    tl.store(B_ptr, a_quantized, mask=mask)

import torch
import triton
import triton.runtime

def quantize_global_transpose(A, BLOCK_M, BLOCK_N, GROUP_M):
    M, N = A.shape
    A = A.contiguous()
    B = torch.empty((N, M), dtype=torch.int8, device=A.device)
    
    # Calculate absmax and its reciprocal
    absmax = torch.max(torch.abs(A)).item()
    absmax_inv = 127.0 / absmax

    # Define grid and block dimensions
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']) // meta['GROUP_M'],
    )

    # Launch the kernel
    _quantize_global_transpose[grid](A, B, absmax_inv, M, N, BLOCK_M, BLOCK_N, GROUP_M,
                                     A.stride(0), A.stride(1), B.stride(0), B.stride(1))

    return B
