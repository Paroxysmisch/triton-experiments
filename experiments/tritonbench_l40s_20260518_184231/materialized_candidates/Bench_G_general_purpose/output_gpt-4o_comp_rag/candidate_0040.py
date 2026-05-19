import torch
import triton
import triton.language as tl

@triton.jit
def _triton_rope(
    Q, K, COS, SIN,  # pointers to matrices
    stride_q_batch, stride_q_seqlen, stride_q_nheads, stride_q_headdim,
    stride_k_batch, stride_k_seqlen, stride_k_nheads, stride_k_headdim,
    seqlen, nheads, rotary_dim, seq_len_ro, BACKWARD_PASS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)

    rotary_dim_half = rotary_dim // 2

    # Calculate starting positions
    q_ptr = Q + pid_batch * stride_q_batch + pid_head * stride_q_nheads
    k_ptr = K + pid_batch * stride_k_batch + pid_head * stride_k_nheads

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk_half = tl.arange(0, BLOCK_K // 2)

    q_ptr = q_ptr + (rm[:, None] * stride_q_seqlen + rk_half[None, :] * stride_q_headdim)
    k_ptr = k_ptr + (rm[:, None] * stride_k_seqlen + rk_half[None, :] * stride_k_headdim)

    COS = COS + (rm[:, None] * rotary_dim_half + rk_half[None, :])
    SIN = SIN + (rm[:, None] * rotary_dim_half + rk_half[None, :])

    cos = tl.load(COS, mask=(rm[:, None] < seq_len_ro) & (rk_half[None, :] < rotary_dim_half), other=1.0).to(tl.float32)
    sin = tl.load(SIN, mask=(rm[:, None] < seq_len_ro) & (rk_half[None, :] < rotary_dim_half), other=0.0).to(tl.float32)

    q0 = tl.load(q_ptr, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0).to(tl.float32)
    q1 = tl.load(q_ptr + rotary_dim_half * stride_q_headdim, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0).to(tl.float32)

    k0 = tl.load(k_ptr, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0).to(tl.float32)
    k1 = tl.load(k_ptr + rotary_dim_half * stride_k_headdim, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0).to(tl.float32)

    if BACKWARD_PASS:
        sin = -sin

    q0_new = q0 * cos - q1 * sin
    q1_new = q0 * sin + q1 * cos

    k0_new = k0 * cos - k1 * sin
    k1_new = k0 * sin + k1 * cos

    tl.store(q_ptr, q0_new, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
    tl.store(q_ptr + rotary_dim_half * stride_q_headdim, q1_new, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))

    tl.store(k_ptr, k0_new, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
    tl.store(k_ptr + rotary_dim_half * stride_k_headdim, k1_new, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))

def rope_forward(q, k, cos, sin, batch_size, seq_len, nheads, rotary_dim, seq_len_ro, backward_pass=False):
    assert q.is_contiguous() and k.is_contiguous(), "Inputs must be contiguous"
    assert cos.shape == sin.shape, "Cosine and sine must have the same shape"
    assert rotary_dim <= q.shape[-1], "Rotary dimension must be less than or equal to head dimension"

    BLOCK_M = 4
    BLOCK_K = 32 if rotary_dim <= 32 else 64 if rotary_dim <= 64 else 128 if rotary_dim <= 128 else 256

    grid = (triton.cdiv(seq_len, BLOCK_M), batch_size, nheads)

    with torch.cuda.device(q.device):
        _triton_rope[grid](
            q, k, cos, sin,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            seq_len, nheads, rotary_dim, seq_len_ro, backward_pass,
            BLOCK_M, BLOCK_K
        )

    return q, k
