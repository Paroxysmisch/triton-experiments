import torch
import triton
import triton.language as tl

# Define constants for block sizes
BLOCK_M = 128
BLOCK_N = 128
BLOCK_DMODEL = 64

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, O, 
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    IS_CAUSAL: tl.constexpr, USE_FP8: tl.constexpr
):
    # Extract program IDs
    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)

    # Calculate offsets for Q, K, V, O based on program IDs
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Compute pointers for Q, K, V, and O
    q_ptrs = Q + pid_hz * stride_qz + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    k_ptrs = K + pid_hz * stride_kz + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
    v_ptrs = V + pid_hz * stride_vz + offs_n[None, :] * stride_vn + offs_d[:, None] * stride_vd
    o_ptrs = O + pid_hz * stride_oz + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od

    # Load Q, K, V blocks
    Q_block = tl.load(q_ptrs)
    K_block = tl.load(k_ptrs)
    V_block = tl.load(v_ptrs)

    # Compute scaled dot-product
    scores = tl.dot(Q_block, K_block) * sm_scale

    # Apply causal mask if needed
    if IS_CAUSAL:
        mask = offs_m[:, None] >= offs_n[None, :]
        scores = tl.where(mask, scores, float('-inf'))

    # Compute softmax over scores
    max_scores = tl.max(scores, axis=1)
    scores = scores - max_scores[:, None]
    exp_scores = tl.exp(scores)
    sum_exp_scores = tl.sum(exp_scores, axis=1)
    softmax_scores = exp_scores / sum_exp_scores[:, None]

    # Compute output by weighted sum of V
    output = tl.dot(softmax_scores, V_block)

    # Store the result
    tl.store(o_ptrs, output)

def triton_fa(Q, K, V, sm_scale, IS_CAUSAL=False, USE_FP8=False):
    batch_size, n_heads, m_size, d_model = Q.shape
    grid = (triton.cdiv(m_size, BLOCK_M), batch_size * n_heads)

    O = torch.empty_like(Q)

    _fwd_kernel[grid](
        Q, K, V, sm_scale, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        IS_CAUSAL=IS_CAUSAL, USE_FP8=USE_FP8,
        num_warps=4 if BLOCK_N <= 64 else 8,
        num_stages=2
    )
    return O
