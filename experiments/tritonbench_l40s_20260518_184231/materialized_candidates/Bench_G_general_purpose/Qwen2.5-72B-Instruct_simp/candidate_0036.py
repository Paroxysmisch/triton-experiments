import triton
import triton.language as tl

@triton.jit
def _score_kernel(
    Q, K, M, Out,
    stride_qm, stride_qk,
    stride_km, stride_kk,
    stride_om, stride_on,
    stride_m, 
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_DMODEL: tl.constexpr,
    scale: tl.constexpr
):
    # Compute program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(Q.shape[0], BLOCK_M)
    num_pid_n = tl.cdiv(K.shape[1], BLOCK_N)
    num_pid_in_block_m = pid % num_pid_m
    num_pid_in_block_n = pid // num_pid_m

    # Compute the block offsets
    offs_m = num_pid_in_block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = num_pid_in_block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # Compute the block pointers
    Q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    K_ptrs = K + (offs_k[:, None] * stride_km + offs_n[None, :] * stride_kk)
    M_ptrs = M + (offs_m[:, None] * stride_m + offs_n[None, :])
    Out_ptrs = Out + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)

    # Load the blocks
    q = tl.load(Q_ptrs)
    k = tl.load(K_ptrs)
    m = tl.load(M_ptrs, mask=offs_m[:, None] < Q.shape[0], other=-float('inf'))

    # Compute the dot product
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, BLOCK_DMODEL, 16):
        qk = q[:, k:k+16]
        kk = k[:, k:k+16]
        acc += tl.dot(qk, kk, trans_b=True)

    # Apply scale and mask
    acc *= scale
    acc += m

    # Store the results
    tl.store(Out_ptrs, acc, mask=offs_m[:, None] < Q.shape[0])

import torch
import triton
import triton.language as tl

def get_score(Q, K, M, Out, scale):
    # Get the shapes of the input tensors
    M, K = Q.shape
    N = K.shape[1]

    # Set the block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = 64

    # Compute the grid size
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    # Set the strides
    stride_qm = Q.stride(0)
    stride_qk = Q.stride(1)
    stride_km = K.stride(0)
    stride_kk = K.stride(1)
    stride_om = Out.stride(0)
    stride_on = Out.stride(1)
    stride_m = M.stride(0) if M is not None else 1

    # Run the kernel
    try:
        _score_kernel[grid](
            Q, K, M, Out,
            stride_qm, stride_qk,
            stride_km, stride_kk,
            stride_om, stride_on,
            stride_m,
            BLOCK_M, BLOCK_N, BLOCK_DMODEL,
            scale
        )
    except triton.OutOfResources as e:
        # If out of resources, reduce block sizes and retry
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_DMODEL = 32
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        _score_kernel[grid](
            Q, K, M, Out,
            stride_qm, stride_qk,
            stride_km, stride_kk,
            stride_om, stride_on,
            stride_m,
            BLOCK_M, BLOCK_N, BLOCK_DMODEL,
            scale
        )

    return Out
