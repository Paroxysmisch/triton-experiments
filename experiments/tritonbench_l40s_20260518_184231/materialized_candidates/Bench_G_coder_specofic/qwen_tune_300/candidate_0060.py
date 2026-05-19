import torch
import triton
import triton.language as tl
from packaging import version

@triton.jit
def _triton_rope(
    q_ptr,
    k_ptr,
    cos,
    sin,
    pid,
    ROTARY_INTERLEAVED,
    BACKWARD_PASS,
    BLOCK_SIZE: tl.constexpr,
    PADDED_SEQLEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # Triton kernel to apply rotary position embeddings
    seq_len = PADDED_SEQLEN
    head_dim = HEAD_DIM
    rotary_dim = head_dim // 2

    # Offsets for pid
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    seq_offsets = offsets % seq_len
    rotary_offsets = tl.arange(0, rotary_dim)

    # Load q and k values
    q1 = tl.load(
        q_ptr
        + seq_offsets[:, None] * head_dim
        + rotary_offsets[None, :],
        mask=seq_offsets[:, None] < seq_len,
        other=0.0,
    )
    q2 = tl.load(
        q_ptr
        + seq_offsets[:, None] * head_dim
        + rotary_offsets[None, :]
        + rotary_dim,
        mask=seq_offsets[:, None] < seq_len,
        other=0.0,
    )
    k1 = tl.load(
        k_ptr
        + seq_offsets[:, None] * head_dim
        + rotary_offsets[None, :],
        mask=seq_offsets[:, None] < seq_len,
        other=0.0,
    )
    k2 = tl.load(
        k_ptr
        + seq_offsets[:, None] * head_dim
        + rotary_offsets[None, :]
        + rotary_dim,
        mask=seq_offsets[:, None] < seq_len,
        other=0.0,
    )

    if not BACKWARD_PASS:
        # Forward pass: apply standard rotary embedding
        q1_out = q1 * cos[None, rotary_offsets] - q2 * sin[None, rotary_offsets]
        q2_out = q2 * cos[None, rotary_offsets] + q1 * sin[None, rotary_offsets]
        k1_out = k1 * cos[None, rotary_offsets] - k2 * sin[None, rotary_offsets]
        k2_out = k2 * cos[None, rotary_offsets] + k1 * sin[None, rotary_offsets]
    else:
        # Backward pass: apply inverse rotary embedding
        q1_out = q1 * cos[None, rotary_offsets] + q2 * sin[None, rotary_offsets]
        q2_out = q2 * cos[None, rotary_offsets] - q1 * sin[None, rotary_offsets]
        k1_out = k1 * cos[None, rotary_offsets] + k2 * sin[None, rotary_offsets]
        k2_out = k2 * cos[None, rotary_offsets] - k1 * sin[None, rotary_offsets]

    # Store the results back to q_ptr and k_ptr
    tl.store(
        q_ptr
        + seq_offsets[:, None] * head_dim
        + rotary_offsets[None, :],
        q1_out,
        mask=seq_offsets[:, None] < seq_len,
    )
    tl.store(
        q_ptr
        + seq_offsets[:, None] * head_dim
        + rotary_offsets[None, :]
        + rotary_dim,
        q2_out,
        mask=seq_offsets[:, None] < seq_len,
    )
    tl.store(
        k_ptr
        + seq_offsets[:, None] * head_dim
        + rotary_offsets[None, :],
        k1_out,
        mask=seq_offsets[:, None] < seq_len,
    )
    tl.store(
        k_ptr
        + seq_offsets[:, None] * head_dim
        + rotary_offsets[None, :]
        + rotary_dim,
        k2_out,
        mask=seq_offsets[:, None] < seq_len,
    )


def rope_backward(
    dq: torch.Tensor,
    dk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
    block_size: int = 64,
):
    # Wrapper for backward pass of the Triton rope kernel
    if dq.stride()[-2:] != (1, None) or dk.stride()[-2:] != (1, None):
        dq = dq.contiguous()
        dk = dk.contiguous()

    batch, seq_len, n_heads, head_dim = dq.shape
    assert head_dim % 2 == 0, f"Head dim {head_dim} must be even"
    padded_seq_len = get_padded_size(seq_len, block_size)

    dq = torch.transpose(dq, 1, 2).reshape(batch * n_heads, seq_len, head_dim)
    dk = torch.transpose(dk, 1, 2).reshape(batch * n_heads, seq_len, head_dim)

    grid = lambda META: (triton.cdiv(seq_len, META["BLOCK_SIZE"]),)
    _triton_rope[grid](
        dq,
        dk,
        cos,
        sin,
        0,
        interleaved,
        True,
        block_size,
        padded_seq_len,
        head_dim,
    )
    dq.copy_(torch.transpose(dq.reshape(batch, n_heads, seq_len, head_dim), 1, 2))
    dk.copy_(torch.transpose(dk.reshape(batch, n_heads, seq_len, head_dim), 1, 2))
    return dq, dk
