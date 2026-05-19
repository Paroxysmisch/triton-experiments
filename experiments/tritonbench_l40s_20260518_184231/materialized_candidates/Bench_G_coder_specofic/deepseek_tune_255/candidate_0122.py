import torch
import triton
import triton.language as tl

@triton.jit
def _rope_embedding(
    Q,
    Q_row_stride,
    cos,
    cos_row_stride,
    sin,
    sin_row_stride,
    seqlen,
    head_dim,
    n_heads,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Triton kernel for computing RoPE
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    cos_ptr = cos + row_idx * cos_row_stride + col_offsets
    sin_ptr = sin + row_idx * sin_row_stride + col_offsets
    q_ptr = Q + row_idx * Q_row_stride + col_offsets

    q = tl.load(q_ptr).to(tl.float32)
    if BACKWARD_PASS:
        q = q * (seqlen * head_dim)
    # split q into two halves along the head_dim
    q1 = tl.arange(0, head_dim // 2)
    q2 = tl.arange(head_dim // 2, head_dim)
    c = tl.load(cos_ptr, mask=q1).to(tl.float32)
    s = tl.load(sin_ptr, mask=q1).to(tl.float32)
    # rotate half
    q2_ptr = q_ptr + head_dim // 2
    q2_rotated = tl.load(q2_ptr, mask=q2).to(tl.float32)
    q2_rotated = q2_rotated * c + tl.sin(q2_rotated) * s
    tl.store(q2_ptr, q2_rotated, mask=q2)
    c = tl.load(cos_ptr, mask=q2).to(tl.float32)
    s = tl.load(sin_ptr, mask=q2).to(tl.float32)
    # rotate half
    q1_ptr = q_ptr + head_dim // 2
    q1_rotated = tl.load(q1_ptr, mask=q1).to(tl.float32)
    q1_rotated = q1_rotated * c + tl.sin(q1_rotated) * s
    tl.store(q1_ptr, q1_rotated, mask=q1)

class Fast_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin):
        # Forward pass for RoPE computation
        batch, seqlen, n_heads, head_dim = Q.shape
        n_rows = Q.numel() // Q.shape[-1]
        BLOCK_SIZE = triton.next_power_of_2(head_dim)
        Q = Q.view(batch * seqlen, n_heads * head_dim)
        Q_row_stride = Q.stride(0)
        cos_row_stride = cos.stride(0)
        sin_row_stride = sin.stride(0)
        _rope_embedding[(n_rows,)](
            Q,
            Q_row_stride,
            cos,
            cos_row_stride,
            sin,
            sin_row_stride,
            seqlen,
            head_dim,
            n_heads,
            False,
            BLOCK_SIZE,
        )
        Q = Q.view(batch, seqlen, n_heads, head_dim)
        ctx.save_for_backward(Q, cos, sin)
        return Q

    @staticmethod
    def backward(ctx, DO):
        # Backward pass for RoPE computation
        Q, cos, sin = ctx.saved_tensors
        return Fast_RoPE_Embedding.apply(DO, cos, sin)

def fast_rope_embedding(Q, K, cos, sin):
    # Apply RoPE transformation to both Q and K
    Q = Fast_RoPE_Embedding.apply(Q, cos, sin)
    K = Fast_RoPE_Embedding.apply(K, cos, sin)
    return Q, K
