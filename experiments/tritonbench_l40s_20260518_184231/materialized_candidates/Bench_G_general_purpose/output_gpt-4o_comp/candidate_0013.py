import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(Q_ptr, K_ptr, V_ptr, Out_ptr,
                sm_scale, Lq, Lk, Lv, D,
                stride_qb, stride_qh, stride_qm, stride_qd,
                stride_kb, stride_kh, stride_kn, stride_kd,
                stride_vb, stride_vh, stride_vn, stride_vd,
                stride_ob, stride_oh, stride_om, stride_od,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, kv_group_num: tl.constexpr):
    # Compute the block indices
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    m_idx = tl.program_id(2) * BLOCK_M

    # Pointers to the start of each batch/head
    Q = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    K = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
    V = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
    Out = Out_ptr + batch_idx * stride_ob + head_idx * stride_oh

    # Load Q
    Q_tile = tl.load(Q + m_idx * stride_qm + tl.arange(0, BLOCK_M)[:, None] * stride_qd)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    # Loop over K/V tiles
    for n_idx in range(0, Lk, BLOCK_N):
        # Load K and V tiles
        K_tile = tl.load(K + n_idx * stride_kn + tl.arange(0, BLOCK_N)[None, :] * stride_kd)
        V_tile = tl.load(V + n_idx * stride_vn + tl.arange(0, BLOCK_N)[None, :] * stride_vd)

        # Compute attention scores
        attn_scores = tl.dot(Q_tile, K_tile.T) * sm_scale

        # Apply softmax
        attn_scores = tl.softmax(attn_scores, axis=1)

        # Weight values by attention scores
        acc += tl.dot(attn_scores, V_tile)

    # Store the result
    tl.store(Out + m_idx * stride_om + tl.arange(0, BLOCK_M)[:, None] * stride_od, acc)

import torch
import math

def context_attention_fwd(Q, K, V, BLOCK_M=None, kv_group_num=1):
    # Dimensions
    B, H, Lq, D = Q.shape
    _, _, Lk, _ = K.shape
    _, _, Lv, _ = V.shape

    # Determine BLOCK_M
    if BLOCK_M is None:
        BLOCK_M = 128 if torch.cuda.get_device_capability()[0] >= 7 else 64

    # Calculate sm_scale
    sm_scale = 1.0 / math.sqrt(D)

    # Prepare output tensor
    Out = torch.empty((B, H, Lq, D), device=Q.device, dtype=Q.dtype)

    # Define grid size
    grid = lambda meta: (B, H, (Lq + meta['BLOCK_M'] - 1) // meta['BLOCK_M'])

    # Launch Triton kernel
    _fwd_kernel[grid](
        Q, K, V, Out,
        sm_scale, Lq, Lk, Lv, D,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_N=64, kv_group_num=kv_group_num
    )

    return Out
