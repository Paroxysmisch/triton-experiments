import triton
import triton.language as tl

@triton.jit
def _score_kernel(
    Q, K, M, Out,
    stride_qm, stride_qk,
    stride_km, stride_kk,
    stride_m,
    stride_om, stride_on,
    nheads, N_CTX,
    sm_scale,
    window_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    num_pid_n = tl.cdiv(N_CTX, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size = num_pid_in_group // num_pid_m
    pid_m = first_pid_m + (pid % num_pid_m)
    pid_n = (pid % group_size) * group_size

    # Offsets for Q and K
    offs_qm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_qk = tl.arange(0, nheads) * stride_qk
    offs_km = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kk = tl.arange(0, nheads) * stride_kk

    # Offsets for M and Out
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Bounds checking
    qk_mask = (offs_qm < N_CTX)[:, None] & (offs_km < N_CTX)[None, :]
    m_mask = (offs_m < N_CTX)[:, None] & (offs_on < N_CTX)[None, :]

    # Load Q and K
    Q_block = tl.load(Q + offs_qm[:, None] * stride_qm + offs_qk[None, :], mask=qk_mask, other=0.0)
    K_block = tl.load(K + offs_km[:, None] * stride_km + offs_kk[None, :], mask=qk_mask, other=0.0)

    # Compute dot product
    qk = tl.dot(Q_block, K_block, trans_b=True)
    qk *= sm_scale

    # Apply mask
    if window_size > 0:
        qk_mask = (offs_qm[:, None] - offs_km[None, :]) < window_size
        qk = tl.where(qk_mask, qk, float('-inf'))

    # Apply mask M
    M_block = tl.load(M + offs_m[:, None] * stride_m + offs_on[None, :], mask=m_mask, other=0.0)
    qk += M_block

    # Reduce along the key dimension
    o = tl.sum(qk, axis=1)

    # Store the result
    tl.store(Out + offs_m * stride_om + offs_on, o, mask=m_mask)

import torch
import triton
import triton.language as tl

def get_score(Q, K, M, Out, nheads, N_CTX, sm_scale, window_size=0, BLOCK_M=128, BLOCK_N=128):
    # Determine grid size
    grid = lambda META: (
        triton.cdiv(N_CTX, META['BLOCK_M']) * triton.cdiv(N_CTX, META['BLOCK_N']) * nheads,
    )

    # Execute the kernel
    try:
        _score_kernel[grid](
            Q, K, M, Out,
            Q.stride(0), Q.stride(1),
            K.stride(0), K.stride(1),
            M.stride(0),
            Out.stride(0), Out.stride(1),
            nheads, N_CTX,
            sm_scale,
            window_size,
            BLOCK_M, BLOCK_N,
        )
    except triton.OutOfResources:
        # If out of resources, reduce block sizes and retry
        BLOCK_M = BLOCK_M // 2
        BLOCK_N = BLOCK_N // 2
        _score_kernel[grid](
            Q, K, M, Out,
            Q.stride(0), Q.stride(1),
            K.stride(0), K.stride(1),
            M.stride(0),
            Out.stride(0), Out.stride(1),
            nheads, N_CTX,
            sm_scale,
            window_size,
            BLOCK_M, BLOCK_N,
        )

    return Out

# Example usage
N_CTX = 1024
nheads = 8
Q = torch.randn((N_CTX, nheads), device='cuda')
K = torch.randn((N_CTX, nheads), device='cuda')
M = torch.randn((N_CTX, N_CTX), device='cuda')
Out = torch.zeros((N_CTX, N_CTX), device='cuda')
sm_scale = 1.0 / (nheads ** 0.5)

Out = get_score(Q, K, M, Out, nheads, N_CTX, sm_scale, window_size=10)
print(Out)
