import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_b0b, stride_b0h, stride_b0n, stride_b0m,
    stride_ob, stride_oh, stride_om, stride_od,
    N_CTX, P_SEQ, BLOCK_M, BLOCK_N, BLOCK_DMODEL, sm_scale, OUT_DTYPE: tl.constexpr
):
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    num_pid_n = tl.cdiv(N_CTX + P_SEQ, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    pid_m = first_pid_m + (pid % num_pid_m)
    pid_n = (pid % num_pid_in_group) // num_pid_m

    # Block bounds
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    q_offsets = tl.arange(0, BLOCK_DMODEL)

    # Offsets for Q, K, V, B0, and Out
    Q_block_ptr = tl.make_block_ptr(
        base=Q, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qd),
        offsets=(offs_m[0], 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K, shape=(N_CTX + P_SEQ, BLOCK_DMODEL), strides=(stride_kn, stride_kd),
        offsets=(offs_n[0], 0), block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V, shape=(N_CTX + P_SEQ, BLOCK_DMODEL), strides=(stride_vn, stride_vd),
        offsets=(offs_n[0], 0), block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0)
    )
    B0_block_ptr = tl.make_block_ptr(
        base=B0, shape=(N_CTX + P_SEQ, BLOCK_M), strides=(stride_b0n, stride_b0m),
        offsets=(offs_n[0], offs_m[0]), block_shape=(BLOCK_N, BLOCK_M), order=(1, 0)
    )
    Out_block_ptr = tl.make_block_ptr(
        base=Out, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_om, stride_od),
        offsets=(offs_m[0], 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0)
    )

    # Load Q, K, V, and B0
    q = tl.load(Q_block_ptr)
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    b0 = tl.load(B0_block_ptr)

    # Compute QK^T
    qk = tl.dot(q, k, trans_b=True) * sm_scale + b0

    # Compute softmax
    qk = tl.softmax(qk, axis=1)

    # Compute attention output
    acc = tl.dot(qk, v)

    # Store the result
    tl.store(Out_block_ptr, acc.to(OUT_DTYPE))

import triton
import torch

def _attention_rel_h_rel_w_kernel_aligned_device(
    Q, K, V, B0, sm_scale, OUT_DTYPE, BLOCK_M=128, BLOCK_N=128, BLOCK_DMODEL=64, num_warps=4, num_stages=3
):
    # Verify shapes and types
    assert Q.shape == K.shape == V.shape, "Q, K, and V must have the same shape"
    assert B0.shape[0] == Q.shape[0] + K.shape[0] - Q.shape[0], "B0 sequence dimension mismatch"
    assert Q.dtype == K.dtype == V.dtype, "Q, K, and V must have the same data type"
    assert B0.dtype == Q.dtype, "B0 must have the same data type as Q, K, and V"

    # Convert to Triton tensors
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    B0 = B0.contiguous()

    # Output tensor
    Out = torch.empty_like(Q, dtype=OUT_DTYPE)

    # Grid and block dimensions
    grid = (
        (Q.shape[0] * Q.shape[1] * (Q.shape[2] + K.shape[2] - Q.shape[2])) // (BLOCK_M * BLOCK_N),
        1, 1
    )

    # Launch the kernel
    _fwd_kernel_aligned[grid](
        Q, K, V, B0, Out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        B0.stride(0), B0.stride(1), B0.stride(2), B0.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        Q.shape[2], K.shape[2] - Q.shape[2], BLOCK_M, BLOCK_N, BLOCK_DMODEL, sm_scale, OUT_DTYPE
    )

    return Out
