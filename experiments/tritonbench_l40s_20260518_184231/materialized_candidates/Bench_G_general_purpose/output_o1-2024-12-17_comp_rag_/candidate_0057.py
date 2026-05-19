import triton
import triton.language as tl
import torch
from .utils import calculate_settings

ROPE_GROUP_SIZE = 4

@triton.jit
def _triton_rope(
    T,                 # Q or K matrix
    T_row_stride,
    cos, cos_row_stride,
    sin, sin_row_stride,
    seqlen,
    head_dim  : tl.constexpr,
    n_heads   : tl.constexpr,
    BACKWARD_PASS : tl.constexpr,
    BLOCK_SIZE    : tl.constexpr,
):
    row_id  = tl.program_id(0)
    group_id = tl.program_id(1)
    col_offsets  = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2

    mask = col_offsets < half_head_dim
    sin_vals = tl.load(sin + (row_id % seqlen)*sin_row_stride + col_offsets, mask=mask, other=0)
    cos_vals = tl.load(cos + (row_id % seqlen)*cos_row_stride + col_offsets, mask=mask, other=0)

    # Invert rotation for backward pass
    if BACKWARD_PASS:
        sin_vals = -sin_vals

    head_start = group_id * ROPE_GROUP_SIZE
    head_end   = min((head_start + ROPE_GROUP_SIZE), n_heads)

    for head_idx in range(head_start, head_end):
        offs_1 = row_id * T_row_stride + head_idx * head_dim + col_offsets
        offs_2 = row_id * T_row_stride + head_idx * head_dim + col_offsets + half_head_dim

        T1 = tl.load(T + offs_1, mask=mask, other=0).to(sin_vals.dtype)
        T2 = tl.load(T + offs_2, mask=mask, other=0).to(sin_vals.dtype)

        out1 = T1 * cos_vals - T2 * sin_vals
        out2 = T2 * cos_vals + T1 * sin_vals

        tl.store(T + offs_1, out1, mask=mask)
        tl.store(T + offs_2, out2, mask=mask)

def rope_forward(Q, K, cos, sin):
    """
    Applies the RoPE forward pass to Q and K using the _triton_rope kernel.
    """
    # Squeeze extra dims if needed
    cos, sin = cos.squeeze(), sin.squeeze()
    # Q -> [batch, seq_len, n_heads, head_dim]
    # Transpose for shape: [batch, seq_len] -> [batch*seq_len, n_heads*head_dim]
    B, S, H, D = Q.shape
    Q_2d = Q.view(B*S, H*D)
    K_2d = K.view(B*S, H*D)

    BLOCK_SIZE, num_warps = calculate_settings(D // 2)
    div, mod = divmod(H, ROPE_GROUP_SIZE)
    n_groups = div + (mod != 0)

    _triton_rope[(B*S, n_groups)](
        Q_2d,  Q_2d.stride(0),
        cos,   cos.stride(0),
        sin,   sin.stride(0),
        S,
        D, H,
        BACKWARD_PASS=False,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    _triton_rope[(B*S, n_groups)](
        K_2d,  K_2d.stride(0),
        cos,   cos.stride(0),
        sin,   sin.stride(0),
        S,
        D, H,
        BACKWARD_PASS=False,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    Q_out = Q_2d.view(B, S, H, D)
    K_out = K_2d.view(B, S, H, D)
    return Q_out, K_out

def rope_backward(dQ, dK, cos, sin):
    """
    Applies the inverse RoPE rotation to dQ and dK for the backward pass
    using the _triton_rope kernel.
    """
    cos, sin = cos.squeeze(), sin.squeeze()
    B, S, H, D = dQ.shape

    dQ_2d = dQ.view(B*S, H*D)
    dK_2d = dK.view(B*S, H*D)

    BLOCK_SIZE, num_warps = calculate_settings(D // 2)
    div, mod = divmod(H, ROPE_GROUP_SIZE)
    n_groups = div + (mod != 0)

    _triton_rope[(B*S, n_groups)](
        dQ_2d, dQ_2d.stride(0),
        cos,   cos.stride(0),
        sin,   sin.stride(0),
        S,
        D, H,
        BACKWARD_PASS=True,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    _triton_rope[(B*S, n_groups)](
        dK_2d, dK_2d.stride(0),
        cos,   cos.stride(0),
        sin,   sin.stride(0),
        S,
        D, H,
        BACKWARD_PASS=True,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    dQ_out = dQ_2d.view(B, S, H, D)
    dK_out = dK_2d.view(B, S, H, D)
    return dQ_out, dK_out
