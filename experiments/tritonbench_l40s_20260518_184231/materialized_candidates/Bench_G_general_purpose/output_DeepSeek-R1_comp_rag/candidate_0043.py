import torch
import triton
import triton.language as tl

@triton.jit
def _triton_rope(
    Q,  # Pointer to the query tensor
    K,  # Pointer to the key tensor
    COS,  # Pointer to the cosine tensor
    SIN,  # Pointer to the sine tensor
    batch_size,
    seq_len,
    n_heads,
    head_dim,
    stride_q_batch,
    stride_q_seq,
    stride_q_head,
    stride_q_headdim,
    stride_k_batch,
    stride_k_seq,
    stride_k_head,
    stride_k_headdim,
    stride_cos_seq,
    stride_cos_headdim,
    stride_sin_seq,
    stride_sin_headdim,
    rotary_dim,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_batch_seq = tl.program_id(1)
    pid_head = tl.program_id(2)

    rotary_dim_half = rotary_dim // 2

    batch_id = pid_batch_seq // seq_len
    seq_id = pid_batch_seq % seq_len

    if batch_id >= batch_size or seq_id >= seq_len or pid_head >= n_heads:
        return

    dim_start = pid_m * BLOCK_HEADDIM
    dims = dim_start + tl.arange(0, BLOCK_HEADDIM)
    mask = dims < rotary_dim_half

    cos_ptrs = COS + seq_id * stride_cos_seq + dims * stride_cos_headdim
    sin_ptrs = SIN + seq_id * stride_sin_seq + dims * stride_sin_headdim

    cos = tl.load(cos_ptrs, mask=mask, other=1.0)
    sin = tl.load(sin_ptrs, mask=mask, other=0.0)

    if BACKWARD_PASS:
        sin = -sin

    q_offset = (batch_id * stride_q_batch + 
                seq_id * stride_q_seq + 
                pid_head * stride_q_head)
    k_offset = (batch_id * stride_k_batch + 
                seq_id * stride_k_seq + 
                pid_head * stride_k_head)

    x0_ptrs = Q + q_offset + dims * stride_q_headdim
    x1_ptrs = Q + q_offset + (dims + rotary_dim_half) * stride_q_headdim
    x0 = tl.load(x0_ptrs, mask=mask, other=0.0)
    x1 = tl.load(x1_ptrs, mask=mask, other=0.0)

    q0 = x0 * cos - x1 * sin
    q1 = x0 * sin + x1 * cos

    tl.store(x0_ptrs, q0, mask=mask)
    tl.store(x1_ptrs, q1, mask=mask)

    x0_k_ptrs = K + k_offset + dims * stride_k_headdim
    x1_k_ptrs = K + k_offset + (dims + rotary_dim_half) * stride_k_headdim
    x0_k = tl.load(x0_k_ptrs, mask=mask, other=0.0)
    x1_k = tl.load(x1_k_ptrs, mask=mask, other=0.0)

    k0 = x0_k * cos - x1_k * sin
    k1 = x0_k * sin + x1_k * cos

    tl.store(x0_k_ptrs, k0, mask=mask)
    tl.store(x1_k_ptrs, k1, mask=mask)

def rope_forward(q, k, cos, sin, seqlen_offset=0, backward=False, inplace=False):
    assert q.shape == k.shape, "Queries and keys must have the same shape"
    assert q.dim() == 4, "Input tensors must be 4-dimensional (batch, seq_len, heads, head_dim)"
    batch_size, seq_len, n_heads, head_dim = q.shape
    rotary_dim = cos.shape[1] * 2
    assert rotary_dim <= head_dim, "rotary_dim must be less than or equal to head_dim"

    if not inplace:
        q = q.contiguous()
        k = k.contiguous()
    else:
        q = q if q.is_contiguous() else q.contiguous()
        k = k if k.is_contiguous() else k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    BLOCK_HEADDIM = 128 if rotary_dim >= 128 else 64
    grid = (
        triton.cdiv(rotary_dim // 2, BLOCK_HEADDIM),
        batch_size * seq_len,
        n_heads,
    )

    _triton_rope[grid](
        q, k, cos, sin,
        batch_size,
        seq_len,
        n_heads,
        head_dim,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        rotary_dim,
        BACKWARD_PASS=backward,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    return q, k, cos, sin
