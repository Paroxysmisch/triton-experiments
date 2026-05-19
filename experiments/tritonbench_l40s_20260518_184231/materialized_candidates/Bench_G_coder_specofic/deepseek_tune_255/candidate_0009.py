import triton
import triton.language as tl
import torch

MAX_FUSED_SIZE = 65536 // 2

def calculate_settings(n):
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError("This layer norm can't be fused, as the size is too large")
    num_warps = min(max(BLOCK_SIZE // 256, 1), 16)
    return BLOCK_SIZE, num_warps

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
    ROPE_GROUP_SIZE: tl.constexpr,
):
    # Triton kernel implementation
    row_block_id = tl.program_id(0)
    group_head_id = tl.program_id(1)

    row_start = row_block_id * BLOCK_SIZE
    row_end = row_start + BLOCK_SIZE
    row_mask = row_start < seqlen

    group_id = group_head_id // ROPE_GROUP_SIZE
    group_size = min(ROPE_GROUP_SIZE, n_heads - group_id * ROPE_GROUP_SIZE)
    head_off = group_id * ROPE_GROUP_SIZE * head_dim
    head_start = head_off
    head_end = head_off + group_size * head_dim
    head_mask = head_off < n_heads * head_dim

    off_q = row_start * Q_row_stride + head_start
    off_cos = row_start * cos_row_stride + head_start
    off_sin = row_start * sin_row_stride + head_start

    if BACKWARD_PASS:
        for head_idx in range(head_start, head_end, head_dim):
            q = tl.load(Q + off_q + head_idx, mask=row_mask & head_mask).to(tl.float32)
            cos_val = tl.load(cos + off_cos + head_idx, mask=row_mask & head_mask).to(tl.float32)
            sin_val = tl.load(sin + off_sin + head_idx, mask=row_mask & head_mask).to(tl.float32)
            q_rot = q * cos_val + (0 if head_idx + head_dim == head_end else q[head_dim:] * sin_val)
            tl.store(Q + off_q + head_idx, q_rot, mask=row_mask & head_mask)
            if head_idx + head_dim != head_end:
                tl.store(Q + off_q + head_idx + head_dim, q[head_dim:] * cos_val - q_rot * sin_val, mask=row_mask & head_mask)
    else:
        for head_idx in range(head_start, head_end, head_dim):
            q = tl.load(Q + off_q + head_idx, mask=row_mask & head_mask).to(tl.float32)
            cos_val = tl.load(cos + off_cos + head_idx, mask=row_mask & head_mask).to(tl.float32)
            sin_val = tl.load(sin + off_sin + head_idx, mask=row_mask & head_mask).to(tl.float32)
            q_rot = q * cos_val - (0 if head_idx + head_dim == head_end else q[head_dim:] * sin_val)
            tl.store(Q + off_q + head_idx, q_rot, mask=row_mask & head_mask)
            if head_idx + head_dim != head_end:
                tl.store(Q + off_q + head_idx + head_dim, q[head_dim:] * cos_val + q_rot * sin_val, mask=row_mask & head_mask)

@torch.no_grad()
def _rope_embedding_forward_impl(Q, cos, sin):
    # Prepare data for forward pass
    Q = Q.view(Q.shape[0], Q.shape[1] * Q.shape[2], Q.shape[3])
    Q = Q.transpose(1, 2).contiguous()
    seqlen, head_dim, n_heads = Q.shape
    BLOCK_SIZE, num_warps = calculate_settings(head_dim)
    n_groups = (n_heads + ROPE_GROUP_SIZE - 1) // ROPE_GROUP_SIZE

    # Launch the Triton kernel
    _rope_embedding[(seqlen + BLOCK_SIZE - 1) // BLOCK_SIZE, n_groups,](
        Q,
        Q.stride(0),
        cos,
        cos.stride(0),
        sin,
        sin.stride(0),
        seqlen,
        head_dim,
        n_heads,
        False,
        BLOCK_SIZE,
        num_warps=num_warps,
        ROPE_GROUP_SIZE=ROPE_GROUP_SIZE,
    )

@torch.no_grad()
def _rope_embedding_backward_impl(dY, cos, sin, n_groups, BLOCK_SIZE, num_warps):
    # Prepare data for backward pass
    dY = dY.view(dY.shape[0], dY.shape[1] * dY.shape[2], dY.shape[3])
    dY = dY.transpose(1, 2).contiguous()
    seqlen, head_dim, n_heads = dY.shape

    # Launch the Triton kernel for backward pass
    _rope_embedding[(seqlen + BLOCK_SIZE - 1) // BLOCK_SIZE, n_groups,](
        dY,
        dY.stride(0),
        cos,
        cos.stride(0),
        sin,
        sin.stride(0),
        seqlen,
        head_dim,
        n_heads,
        True,
        BLOCK_SIZE,
        num_warps=num_warps,
        ROPE_GROUP_SIZE=ROPE_GROUP_SIZE,
    )
