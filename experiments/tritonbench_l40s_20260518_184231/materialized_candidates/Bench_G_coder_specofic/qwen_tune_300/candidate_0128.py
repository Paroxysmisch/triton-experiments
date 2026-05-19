import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def rotary_kernel(
    X,
    OUT,
    COS,
    SIN,
    CU_SEQLENS: tl.constexpr,
    SEQLEN_OFFSETS,
    INTERLEAVED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MUL_FREQ: tl.constexpr,
    CONJUGATE: tl.constexpr,
    N_CTX: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_BATCHED: tl.constexpr,
    HEAD_DIM_IS_128: tl.constexpr,
    ROTARY_INTERLEAVED_PADDED: tl.constexpr,
    pid_batch,
    pid_head,
    pid_m,
):
    """
    Args:
        X: (batch, head, seq, 2 * head_dim)
        OUT: (batch, head, seq, 2 * head_dim)
        COS: (1, 1, 1, head_dim)
        SIN: (1, 1, 1, head_dim)
        CU_SEQLENS: (batch + 1,) or None
        SEQLEN_OFFSETS: (batch,) or None
        INTERLEAVED: (bool)
        BLOCK_SIZE: (int)
        MUL_FREQ: (bool)
        CONJUGATE: (bool)
        N_CTX: (int)
        IS_VARLEN: (bool)
        IS_BATCHED: (bool)
        HEAD_DIM_IS_128: (bool)
        ROTARY_INTERLEAVED_PADDED: (bool)
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    dim_m = (pid_m * BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE)
    dim_m_mask = dim_m < N_CTX

    if not IS_BATCHED:
        pid_batch = 0

    if IS_VARLEN:
        seq_len_offset = tl.load(SEQLEN_OFFSETS + pid_batch)
        cu_seq_lens = tl.load(CU_SEQLENS + pid_batch)
        cu_seq_lens_next = tl.load(CU_SEQLENS + pid_batch + 1)
        seq_len = cu_seq_lens_next - cu_seq_lens
        rotary_interleaved_padded = ROTARY_INTERLEAVED_PADDED and seq_len < N_CTX
    else:
        seq_len_offset = 0
        seq_len = N_CTX
        rotary_interleaved_padded = False

    if not INTERLEAVED:
        dim_k = tl.arange(0, 2 * BLOCK_SIZE)
        dim_k_mask = dim_k < 2 * HEAD_DIM
        dim_k_half_mask = dim_k < BLOCK_SIZE
        head_dim_mask = None
    else:
        half_block_size = BLOCK_SIZE // 2
        dim_k = ((half_block_size + tl.arange(0, half_block_size)) * 2) % (2 * HEAD_DIM)
        dim_k_mask = dim_k < 2 * HEAD_DIM
        dim_k_half_mask = (dim_k < HEAD_DIM) | (dim_k >= half_block_size)
        head_dim_mask = tl.arange(0, half_block_size) < HEAD_DIM

    X_PTR = X + (pid_batch * HEADS * N_CTX * 2) + (pid_head * N_CTX * 2) + (
        seq_len_offset * 2
    ) + (dim_m[:, None] * 2) + dim_k[None, :]
    if OUT is X:
        X_PTR = tl.make_block_ptr(
            base=X,
            shape=(N_CTX, 2 * HEAD_DIM),
            strides=(2 * HEAD_DIM, 1),
            offsets=(seq_len_offset + dim_m[None, :] * 2, 0),
            block_shape=(BLOCK_SIZE, 2),
            order=(1, 0),
        )
    OUT_PTR = OUT + (pid_batch * HEADS * N_CTX * 2) + (pid_head * N_CTX * 2) + (
        seq_len_offset * 2
    ) + (dim_m[:, None] * 2) + dim_k[None, :]
    if CONJUGATE:
        COS_PTR = COS + pid_head * HEAD_DIM + (dim_k // 2) + (BLOCK_SIZE * HEAD_DIM)
        SIN_PTR = SIN + pid_head * HEAD_DIM + (dim_k // 2) + (BLOCK_SIZE * HEAD_DIM)
    else:
        COS_PTR = COS + pid_head * HEAD_DIM + (dim_k // 2)
        SIN_PTR = SIN + pid_head * HEAD_DIM + (dim_k // 2)

    if not HEAD_DIM_IS_128:
        COS_VAL = tl.load(COS_PTR, mask=dim_k_mask, other=0.0).to(tl.float32)
        SIN_VAL = tl.load(SIN_PTR, mask=dim_k_mask, other=0.0).to(tl.float32)
    else:
        COS_VAL = tl.load(COS_PTR, mask=head_dim_mask, other=0.0).to(tl.float32)
        SIN_VAL = tl.load(SIN_PTR, mask=head_dim_mask, other=0.0).to(tl.float32)

    if not INTERLEAVED:
        if not HEAD_DIM_IS_128:
            X0 = tl.load(X_PTR, mask=dim_m_mask[:, None] & dim_k_mask[None, :], other=0.0)
            X1 = tl.load(
                X_PTR + HEAD_DIM, mask=dim_m_mask[:, None] & dim_k_mask[None, :], other=0.0
            )
        else:
            X0 = tl.load(X_PTR, mask=dim_m_mask[:, None] & head_dim_mask[None, :], other=0.0)
            X1 = tl.load(
                X_PTR + HEAD_DIM, mask=dim_m_mask[:, None] & head_dim_mask[None, :], other=0.0
            )
    else:
        if not HEAD_DIM_IS_128:
            X0 = tl.load(
                X_PTR, mask=dim_m_mask[:, None] & dim_k_half_mask[None, :], other=0.0
            )
            X1 = tl.load(
                X_PTR + HEAD_DIM, mask=dim_m_mask[:, None] & dim_k_half_mask[None, :], other=0.0
            )
        else:
            X0 = tl.load(
                X_PTR, mask=dim_m_mask[:, None] & dim_k_half_mask[None, :], other=0.0
            )
            X1 = tl.load(
                X_PTR + HEAD_DIM, mask=dim_m_mask[:, None] & dim_k_half_mask[None, :], other=0.0
            )

    if MUL_FREQ:
        if not INTERLEAVED:
            COS_VAL = tl.load(COS + pid_head * HEAD_DIM + dim_k).to(tl.float32)
            SIN_VAL = tl.load(SIN + pid_head * HEAD_DIM + dim_k).to(tl.float32)
        else:
            COS_VAL = tl.load(COS + pid_head * HEAD_DIM + (dim_k // 2)).to(tl.float32)
            SIN_VAL = tl.load(SIN + pid_head * HEAD_DIM + (dim_k // 2)).to(tl.float32)

    if INTERLEAVED:
        X0_NEXT = tl.load(
            X + (pid_batch * HEADS * N_CTX * 2)
            + (pid_head * N_CTX * 2)
            + (seq_len_offset * 2)
            + ((dim_m + 1) % N_CTX * 2)
            + dim_k[None, :],
            mask=dim_m_mask[:, None] & dim_k_mask[None, :],
            other=0.0,
        )
        if CONJUGATE:
            X0_NEXT = -X0_NEXT
        X0 = X0 + 1j * X0_NEXT
    else:
        if not HEAD_DIM_IS_128:
            X1_NEXT = tl.load(
                X + (pid_batch * HEADS * N_CTX * 2)
                + (pid_head * N_CTX * 2)
                + (seq_len_offset * 2)
                + dim_m[:, None] * 2
                + ((dim_k + 1) % (2 * HEAD_DIM)),
                mask=dim_m_mask[:, None] & dim_k_mask[None, :],
                other=0.0,
            )
        else:
            X1_NEXT = tl.load(
                X + (pid_batch * HEADS * N_CTX * 2)
                + (pid_head * N_CTX * 2)
                + (seq_len_offset * 2)
                + dim_m[:, None] * 2
                + ((dim_k + 1) % (2 * HEAD_DIM)),
                mask=dim_m_mask[:, None] & head_dim_mask[None, :],
                other=0.0,
            )
        if CONJUGATE:
            X1_NEXT = -X1_NEXT
        X1 = X1 + 1j * X1_NEXT

    if ROTARY_INTERLEAVED_PADDED:
        X0 = tl.where((dim_m[:, None] < seq_len) & dim_k_mask[None, :], X0, 0.0)
        X1 = tl.where((dim_m[:, None] < seq_len) & dim_k_mask[None, :], X1, 0.0)

    if HEAD_DIM_IS_128:
        OUT0 = X0 * COS_VAL - X1 * SIN_VAL
