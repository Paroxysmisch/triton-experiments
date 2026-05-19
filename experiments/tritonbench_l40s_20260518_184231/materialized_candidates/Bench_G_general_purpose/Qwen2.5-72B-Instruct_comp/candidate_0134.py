import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    sm_scale,
    B_Start_Loc, B_Seqlen,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(B_Seqlen, BLOCK_M)
    num_pid_n = tl.cdiv(B_Seqlen, BLOCK_N)
    num_pid_in_batch = num_pid_m * num_pid_n
    batch_id = pid // num_pid_in_batch
    pid %= num_pid_in_batch
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Compute the offsets for Q, K, V, and Out
    off_qm = (B_Start_Loc[batch_id] + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) * stride_qd
    off_kn = (B_Start_Loc[batch_id] + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) * stride_kd
    off_v = (B_Start_Loc[batch_id] + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) * stride_vd
    off_o = (B_Start_Loc[batch_id] + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) * stride_od

    # Initialize pointers for Q, K, V, and Out
    q_ptrs = Q + batch_id * stride_qb + off_qm[:, None]
    k_ptrs = K + batch_id * stride_kb + off_kn[None, :]
    v_ptrs = V + batch_id * stride_vb + off_v[None, :]
    o_ptrs = Out + batch_id * stride_ob + off_o[:, None]

    # Load Q and K
    q = tl.load(q_ptrs)
    k = tl.load(k_ptrs)

    # Compute attention scores
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for p in range(0, B_Seqlen, BLOCK_N):
        k_block_ptr = k_ptrs + p * stride_kd
        v_block_ptr = v_ptrs + p * stride_vd
        k_block = tl.load(k_block_ptr)
        qk = tl.dot(q, k_block, allow_tf32=False)
        qk *= sm_scale
        acc += qk

    # Apply softmax
    acc = tl.softmax(acc, axis=1)

    # Load V and compute the output
    v = tl.load(v_ptrs)
    o = tl.dot(acc, v, allow_tf32=False)

    # Store the output
    tl.store(o_ptrs, o)

import torch
import triton
import triton.language as tl

def context_attention_fwd(Q, K, V, sm_scale, B_Start_Loc, B_Seqlen):
    # Get tensor shapes
    B, H, Lq, D = Q.shape
    Lk = K.shape[2]
    Lv = V.shape[2]

    # Allocate output tensor
    Out = torch.empty((B, H, Lq, D), device=Q.device, dtype=Q.dtype)

    # Define grid and block sizes
    BLOCK_M = 128
    BLOCK_DMODEL = D
    BLOCK_N = 128
    grid = (B * H * (Lq // BLOCK_M) * (Lk // BLOCK_N),)

    # Compute the softmax scaling factor
    sm_scale = 1.0 / (D ** 0.5)

    # Launch the kernel
    _fwd_kernel[grid](
        Q, K, V, Out,
        sm_scale,
        B_Start_Loc, B_Seqlen,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        BLOCK_M, BLOCK_DMODEL, BLOCK_N
    )

    return Out
