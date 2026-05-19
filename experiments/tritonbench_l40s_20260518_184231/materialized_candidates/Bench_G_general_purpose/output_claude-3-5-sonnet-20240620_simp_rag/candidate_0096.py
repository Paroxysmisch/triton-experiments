import triton
import triton.language as tl
import torch

@triton.jit
def _rotary_kernel(
    Q, K, Cos, Sin,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_cos_s, stride_cos_d,
    stride_sin_s, stride_sin_d,
    max_total_len, HEAD_Q, HEAD_K,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    pid_h = tl.program_id(0)
    pid_s = tl.program_id(1)

    # Compute the offsets for this block
    offset_h = pid_h * BLOCK_HEAD
    offset_s = pid_s * BLOCK_SEQ

    # Load Q and K
    q_ptr = Q + offset_h * stride_q_h + offset_s * stride_q_s
    k_ptr = K + offset_h * stride_k_h + offset_s * stride_k_s

    # Load Cos and Sin
    cos_ptr = Cos + offset_s * stride_cos_s
    sin_ptr = Sin + offset_s * stride_sin_s

    # Create masks for boundary checks
    mask_h = offset_h < HEAD_Q
    mask_s = tl.arange(0, BLOCK_SEQ) + offset_s < max_total_len

    for d in range(0, BLOCK_DMODEL, BLOCK_DMODEL):
        # Load Q and K blocks
        q = tl.load(q_ptr + d, mask=mask_h & mask_s[:, None], other=0.0)
        k = tl.load(k_ptr + d, mask=mask_h & mask_s[:, None], other=0.0)

        # Load Cos and Sin blocks
        cos = tl.load(cos_ptr + d, mask=mask_s[:, None], other=1.0)
        sin = tl.load(sin_ptr + d, mask=mask_s[:, None], other=0.0)

        # Apply rotary embedding
        q_rot = q * cos - tl.roll(q, 1, 1) * sin
        k_rot = k * cos - tl.roll(k, 1, 1) * sin

        # Store rotated Q and K
        tl.store(q_ptr + d, q_rot, mask=mask_h & mask_s[:, None])
        tl.store(k_ptr + d, k_rot, mask=mask_h & mask_s[:, None])

def rotary_emb_fwd(q, k, cos, sin):
    assert q.shape == k.shape
    assert q.dtype == k.dtype
    assert cos.shape == sin.shape
    assert cos.dtype == q.dtype and sin.dtype == q.dtype

    batch_size, num_heads, seq_len, dim = q.shape
    max_total_len = seq_len

    # Compute strides
    stride_q_b, stride_q_h, stride_q_s, stride_q_d = q.stride()
    stride_k_b, stride_k_h, stride_k_s, stride_k_d = k.stride()
    stride_cos_s, stride_cos_d = cos.stride()
    stride_sin_s, stride_sin_d = sin.stride()

    # Define block sizes
    BLOCK_HEAD = triton.next_power_of_2(num_heads)
    BLOCK_SEQ = 32
    BLOCK_DMODEL = 64

    # Determine grid size
    grid = (triton.cdiv(num_heads, BLOCK_HEAD), triton.cdiv(seq_len, BLOCK_SEQ))

    # Determine number of warps
    num_warps = 4 if dim > 64 else 2

    # Launch kernel
    _rotary_kernel[grid](
        q, k, cos, sin,
        stride_q_b, stride_q_h, stride_q_s, stride_q_d,
        stride_k_b, stride_k_h, stride_k_s, stride_k_d,
        stride_cos_s, stride_cos_d,
        stride_sin_s, stride_sin_d,
        max_total_len, num_heads, num_heads,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps
    )

    return q, k
