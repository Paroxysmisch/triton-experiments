import triton
import triton.language as tl
import torch
from .utils import calculate_settings

ROPE_GROUP_SIZE = 4

@triton.jit
def _triton_rope(
    q_ptr, q_row_stride,
    k_ptr, k_row_stride,
    cos, cos_row_stride,
    sin, sin_row_stride,
    seqlen,
    head_dim      : tl.constexpr,
    n_heads       : tl.constexpr,
    BACKWARD_PASS : tl.constexpr,
    BLOCK_SIZE    : tl.constexpr,
):
    """
        Applies Rotary Positional Embedding (RoPE) to query and key matrices.
        RoPE is Q * cos + rotate_half(Q) * sin
    """
    row_position  = tl.program_id(0)
    group_head_position = tl.program_id(1)
    col_offsets  = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    sin1 = tl.load(sin + (row_position % seqlen)*sin_row_stride + \
                   half_head_dim*0 + col_offsets, mask = mask, other = 0)
    cos1 = tl.load(cos + (row_position % seqlen)*cos_row_stride + \
                   half_head_dim*0 + col_offsets, mask = mask, other = 0)

    if BACKWARD_PASS:
        sin1 = -sin1

    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = min((head_start + ROPE_GROUP_SIZE), n_heads)

    for k in range(head_start, head_end):
        offs_q1 = row_position * q_row_stride + k * head_dim + col_offsets
        offs_q2 = row_position * q_row_stride + k * head_dim + col_offsets + half_head_dim

        offs_k1 = row_position * k_row_stride + k * head_dim + col_offsets
        offs_k2 = row_position * k_row_stride + k * head_dim + col_offsets + half_head_dim

        Q1 = tl.load(q_ptr + offs_q1, mask = mask, other = 0).to(sin1.dtype)
        Q2 = tl.load(q_ptr + offs_q2, mask = mask, other = 0).to(sin1.dtype)

        K1 = tl.load(k_ptr + offs_k1, mask = mask, other = 0).to(sin1.dtype)
        K2 = tl.load(k_ptr + offs_k2, mask = mask, other = 0).to(sin1.dtype)

        tl.store(q_ptr + offs_q1, Q1*cos1 - Q2*sin1, mask = mask)
        tl.store(q_ptr + offs_q2, Q2*cos1 + Q1*sin1, mask = mask)

        tl.store(k_ptr + offs_k1, K1*cos1 - K2*sin1, mask = mask)
        tl.store(k_ptr + offs_k2, K2*cos1 + K1*sin1, mask = mask)

class Fast_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, cos, sin):
        cos, sin = cos.squeeze(), sin.squeeze()
        batch, seq_len, n_heads, head_dim = Q.shape
        Q = Q.view(batch*seq_len, n_heads*head_dim)
        K = K.view(batch*seq_len, n_heads*head_dim)
        n_rows, n_cols = Q.shape
        assert(seq_len <= cos.shape[0])

        BLOCK_SIZE, num_warps = calculate_settings(head_dim//2)
        
        div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
        n_groups = div + (mod != 0)

        _triton_rope[(n_rows, n_groups, )](
              Q,   Q.stride(0),
              K,   K.stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seq_len,
            head_dim, n_heads,
            BACKWARD_PASS = False,
            BLOCK_SIZE = BLOCK_SIZE,
            num_warps  = num_warps,
        )
        ctx.save_for_backward(Q, K, cos, sin)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps  = num_warps
        ctx.n_groups = n_groups
        return Q.view(batch, seq_len, n_heads, head_dim), K.view(batch, seq_len, n_heads, head_dim)

    @staticmethod
    def backward(ctx, dQ, dK):
        Q, K, cos, sin = ctx.saved_tensors
        batch, seq_len, n_heads, head_dim = dQ.shape
        dQ = dQ.reshape(batch*seq_len, n_heads*head_dim)
        dK = dK.reshape(batch*seq_len, n_heads*head_dim)
        n_rows, n_cols = dQ.shape

        _triton_rope[(n_rows, ctx.n_groups, )](
            dQ,  dQ.stride(0),
            dK,  dK.stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seq_len, head_dim, n_heads,
            BACKWARD_PASS = True,
            BLOCK_SIZE = ctx.BLOCK_SIZE,
            num_warps  = ctx.num_warps,
        )
        dQ = dQ.view(batch, seq_len, n_heads, head_dim)
        dK = dK.view(batch, seq_len, n_heads, head_dim)
        return dQ, dK, None, None

def rope_backward(dQ, dK, cos, sin):
    dQ = Fast_RoPE_Embedding.apply(dQ.transpose(1, 2), cos, sin).transpose(1, 2)
    dK = Fast_RoPE_Embedding.apply(dK.transpose(1, 2), cos, sin).transpose(1, 2)
    return dQ, dK

def fast_rope_embedding(Q, K, cos, sin):
    Q, K = Fast_RoPE_Embedding.apply(Q.transpose(1, 2), K.transpose(1, 2), cos, sin)
    return Q.transpose(1, 2), K.transpose(1, 2)
