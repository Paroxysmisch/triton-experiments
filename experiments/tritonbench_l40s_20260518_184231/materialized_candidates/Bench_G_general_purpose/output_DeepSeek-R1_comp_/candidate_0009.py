import torch
import triton
import triton.language as tl

MAX_FUSED_SIZE = 4096
ROPE_GROUP_SIZE = 4  # can be tuned for specific architectures

def calculate_settings(n):
    BLOCK_SIZE = 1
    while BLOCK_SIZE < n:
        BLOCK_SIZE *= 2
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"head_dim {n} exceeds maximum supported fused size {MAX_FUSED_SIZE}")
    if BLOCK_SIZE <= 256:
        num_warps = 4
    elif BLOCK_SIZE <= 2048:
        num_warps = 8
    else:
        num_warps = 16
    return BLOCK_SIZE, num_warps

@triton.jit
def _rope_embedding(
    Q, Q_row_stride, cos, cos_row_stride, sin, sin_row_stride,
    seqlen, head_dim, n_heads, BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, ROPE_GROUP_SIZE: tl.constexpr
):
    pid_row = tl.program_id(0)
    pid_group = tl.program_id(1)
    row = pid_row
    group = pid_group

    h_start = group * ROPE_GROUP_SIZE
    h_end = h_start + ROPE_GROUP_SIZE
    if h_start >= n_heads:
        return

    seq_idx = row % seqlen
    cos_row = cos + seq_idx * cos_row_stride
    sin_row = sin + seq_idx * sin_row_stride

    half_dim = head_dim // 2

    for h_offset in range(ROPE_GROUP_SIZE):
        h = h_start + h_offset
        if h >= n_heads:
            break
        q_ptr = Q + h * Q_row_stride + row * head_dim

        x_block_start = 0
        while x_block_start < half_dim:
            x_offs = x_block_start + tl.arange(0, BLOCK_SIZE)
            mask = x_offs < half_dim

            x_ptr = q_ptr + x_offs
            y_ptr = q_ptr + x_offs + half_dim

            x = tl.load(x_ptr, mask=mask, other=0.0)
            y = tl.load(y_ptr, mask=mask, other=0.0)

            cos_vals = tl.load(cos_row + x_offs, mask=mask, other=0.0)
            sin_vals = tl.load(sin_row + x_offs, mask=mask, other=0.0)

            if BACKWARD_PASS:
                new_x = x * cos_vals + y * sin_vals
                new_y = -x * sin_vals + y * cos_vals
            else:
                new_x = x * cos_vals - y * sin_vals
                new_y = x * sin_vals + y * cos_vals

            tl.store(x_ptr, new_x, mask=mask)
            tl.store(y_ptr, new_y, mask=mask)

            x_block_start += BLOCK_SIZE

def _rope_embedding_forward_impl(Q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    batch, seqlen, n_heads, head_dim = Q.shape
    Q = Q.reshape(batch * seqlen, n_heads, head_dim)
    Q = Q.transpose(0, 1).contiguous()  # [n_heads, batch*seqlen, head_dim]
    n_rows = Q.shape[1]

    BLOCK_SIZE, num_warps = calculate_settings(head_dim)
    n_groups = (n_heads + ROPE_GROUP_SIZE - 1) // ROPE_GROUP_SIZE

    assert cos.shape == (seqlen, head_dim)
    assert sin.shape == (seqlen, head_dim)

    grid = (n_rows, n_groups)
    _rope_embedding[grid](
        Q, Q.stride(1),
        cos, cos.stride(0),
        sin, sin.stride(0),
        seqlen, head_dim, n_heads, False,
        BLOCK_SIZE, ROPE_GROUP_SIZE,
        num_warps=num_warps
    )

    Q = Q.transpose(0, 1).reshape(batch, seqlen, n_heads, head_dim)
    return Q

def _rope_embedding_backward_impl(dY: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, n_groups: int, BLOCK_SIZE: int, num_warps: int):
    batch, seqlen, n_heads, head_dim = dY.shape
    dY = dY.reshape(batch * seqlen, n_heads, head_dim)
    dY = dY.transpose(0, 1).contiguous()
    n_rows = dY.shape[1]

    assert cos.shape == (seqlen, head_dim)
    assert sin.shape == (seqlen, head_dim)

    grid = (n_rows, n_groups)
    _rope_embedding[grid](
        dY, dY.stride(1),
        cos, cos.stride(0),
        sin, sin.stride(0),
        seqlen, head_dim, n_heads, True,
        BLOCK_SIZE, ROPE_GROUP_SIZE,
        num_warps=num_warps
    )

    dY = dY.transpose(0, 1).reshape(batch, seqlen, n_heads, head_dim)
    return dY

class RoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin):
        Q = _rope_embedding_forward_impl(Q, cos, sin)
        ctx.save_for_backward(cos, sin)
        ctx.BLOCK_SIZE, ctx.num_warps = calculate_settings(Q.size(-1))
        ctx.n_groups = (Q.size(2) + ROPE_GROUP_SIZE - 1) // ROPE_GROUP_SIZE
        return Q

    @staticmethod
    def backward(ctx, dY):
        cos, sin = ctx.saved_tensors
        dY = _rope_embedding_backward_impl(dY, cos, sin, ctx.n_groups, ctx.BLOCK_SIZE, ctx.num_warps)
        return dY, None, None

def rope_embedding(Q, cos, sin):
    return RoPEFunction.apply(Q, cos, sin)
