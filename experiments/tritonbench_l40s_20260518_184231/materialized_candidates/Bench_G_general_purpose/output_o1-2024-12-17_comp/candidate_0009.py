import math
import torch
import triton
import triton.language as tl

MAX_FUSED_SIZE = 2048

def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()

def calculate_settings(n: int) -> (int, int):
    block_size = _next_power_of_2(n)
    if block_size > MAX_FUSED_SIZE:
        raise RuntimeError("Block size too large!")
    num_warps = 4 if block_size >= 1024 else 2
    return block_size, num_warps

@triton.jit
def _rope_embedding(
    Q, Q_row_stride,
    cos, cos_row_stride,
    sin, sin_row_stride,
    seqlen, head_dim, n_heads,
    BACKWARD_PASS,
    BLOCK_SIZE: tl.constexpr,
    ROPE_GROUP_SIZE: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    idx = tl.arange(0, BLOCK_SIZE)
    row = pid_m * BLOCK_SIZE + idx
    mask = row < seqlen

    group_offset = pid_n * head_dim * ROPE_GROUP_SIZE
    q_ptr = Q + row * Q_row_stride + group_offset
    c_ptr = cos + row * cos_row_stride + group_offset
    s_ptr = sin + row * sin_row_stride + group_offset

    D = head_dim
    col = tl.arange(0, D, dtype=tl.int32)

    halfD = (D // 2)
    half_mask = col < halfD

    for head_idx in range(ROPE_GROUP_SIZE):
        q_off = q_ptr + head_idx * D
        c_off = c_ptr + head_idx * D
        s_off = s_ptr + head_idx * D

        q_val = tl.load(q_off + col, mask=mask & (col < D), other=0.0)
        c_val = tl.load(c_off + col, mask=mask & (col < D), other=0.0)
        s_val = tl.load(s_off + col, mask=mask & (col < D), other=0.0)

        q_left = tl.where(half_mask, q_val, 0.0)
        q_right = tl.where(~half_mask, q_val, 0.0)
        c_left = tl.where(half_mask, c_val, 0.0)
        c_right = tl.where(~half_mask, c_val, 0.0)
        s_left = tl.where(half_mask, s_val, 0.0)
        s_right = tl.where(~half_mask, s_val, 0.0)

        if not BACKWARD_PASS:
            left = q_left * c_left - q_right * s_left
            right = q_right * c_right + q_left * s_right
        else:
            left = q_left * c_left + q_right * s_left
            right = q_right * c_right - q_left * s_right

        out_left = tl.where(half_mask, left, 0.0)
        out_right = tl.where(~half_mask, right, 0.0)
        new_q_val = out_left + out_right

        tl.store(q_off + col, new_q_val, mask=mask & (col < D))

@torch.no_grad()
def _rope_embedding_forward_impl(Q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, seqlen, n_heads, head_dim = Q.shape
    Q_flat = Q.reshape(B * n_heads, seqlen, head_dim)
    Q_row_stride = head_dim

    cos_flat = cos.reshape(seqlen, n_heads * head_dim)
    sin_flat = sin.reshape(seqlen, n_heads * head_dim)
    cos_row_stride = n
