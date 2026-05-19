import torch
import triton
import triton.language as tl

@triton.jit
def _rotary_kernel(
    Q, K, Cos, Sin,
    stride_q_b, stride_q_h, stride_q_l, stride_q_d,
    stride_k_b, stride_k_h, stride_k_l, stride_k_d,
    stride_cos_l, stride_cos_d,
    stride_sin_l, stride_sin_d,
    B, H, L, D,
    BLOCK_HEAD: tl.constexpr, BLOCK_SEQ: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    pid_heade = tl.program_id(0)
    pid_seq = tl.program_id(1)

    head_start = pid_heade * BLOCK_HEAD
    seq_start = pid_seq * BLOCK_SEQ

    head_offsets = head_start + tl.arange(0, BLOCK_HEAD)
    seq_offsets = seq_start + tl.arange(0, BLOCK_SEQ)

    head_mask = head_offsets < H
    seq_mask = seq_offsets < L

    d_offset = tl.arange(0, BLOCK_DMODEL)
    d_mask = d_offset < D
    cos_sin_idx = d_offset // 2
    d_pair_mask = cos_sin_idx < (D // 2)

    for b in range(B):
        for d in range(0, D, BLOCK_DMODEL):
            current_d = d + d_offset
            current_d_mask = d_mask & (current_d < D)
            current_cos_sin_idx = cos_sin_idx + (d // 2)
            current_cos_sin_mask = d_pair_mask & (current_cos_sin_idx < (D // 2))

            cos_ptrs = Cos + (seq_offsets[:, None] * stride_cos_l + current_cos_sin_idx[None, :] * stride_cos_d)
            sin_ptrs = Sin + (seq_offsets[:, None] * stride_sin_l + current_cos_sin_idx[None, :] * stride_sin_d)

            cos = tl.load(cos_ptrs, mask=seq_mask[:, None] & current_cos_sin_mask[None, :], other=0.0)
            sin = tl.load(sin_ptrs, mask=seq_mask[:, None] & current_cos_sin_mask[None, :], other=0.0)

            q_ptrs = (
                Q + b * stride_q_b +
                head_offsets[:, None, None] * stride_q_h +
                seq_offsets[None, :, None] * stride_q_l +
                (current_d[None, None, :] * stride_q_d)
            )
            k_ptrs = (
                K + b * stride_k_b +
                head_offsets[:, None, None] * stride_k_h +
                seq_offsets[None, :, None] * stride_k_l +
                (current_d[None, None, :] * stride_k_d)
            )

            q = tl.load(q_ptrs, mask=head_mask[:, None, None] & seq_mask[None, :, None] & current_d_mask[None, None, :], other=0.0)
            k = tl.load(k_ptrs, mask=head_mask[:, None, None] & seq_mask[None, :, None] & current_d_mask[None, None, :], other=0.0)

            q0 = q[:, :, 0::2]
            q1 = q[:, :, 1::2]
            k0 = k[:, :, 0::2]
            k1 = k[:, :, 1::2]

            new_q0 = q0 * cos - q1 * sin
            new_q1 = q0 * sin + q1 * cos
            new_k0 = k0 * cos - k1 * sin
            new_k1 = k0 * sin + k1 * cos

            new_q = tl.zeros_like(q)
            new_q = tl.view(new_q, (*new_q.shape[:-1], new_q.shape[-1]//2, 2))
            new_q = tl.view(new_q.at[:, :, :, 0].set(new_q0), new_q.shape)
            new_q = tl.view(new_q.at[:, :, :, 1].set(new_q1), new_q.shape)
            new_q = tl.reshape(new_q, (*new_q.shape[:2], D))

            new_k = tl.zeros_like(k)
            new_k = tl.view(new_k, (*new_k.shape[:-1], new_k.shape[-1]//2, 2))
            new_k = tl.view(new_k.at[:, :, :, 0].set(new_k0), new_k.shape)
            new_k = tl.view(new_k.at[:, :, :, 1].set(new_k1), new_k.shape)
            new_k = tl.reshape(new_k, (*new_k.shape[:2], D))

            tl.store(q_ptrs, new_q, mask=head_mask[:, None, None] & seq_mask[None, :, None] & current_d_mask[None, None, :])
            tl.store(k_ptrs, new_k, mask=head_mask[:, None, None] & seq_mask[None, :, None] & current_d_mask[None, None, :])

def rotary_emb_fwd(q, k, cos, sin):
    B, H, L, D = q.shape
    assert k.shape == q.shape, "Q and K must have the same shape"
    assert cos.shape == (L, D//2), f"Cos shape {cos.shape} does not match {(L, D//2)}"
    assert sin.shape == (L, D//2), f"Sin shape {sin.shape} does not match {(L, D//2)}"

    BLOCK_HEAD = 4
    BLOCK_SEQ = 32
    BLOCK_DMODEL = 64

    grid = (triton.cdiv(H, BLOCK_HEAD), triton.cdiv(L, BLOCK_SEQ))
    num_warps = 4 if BLOCK_DMODEL >= 64 else 2

    _rotary_kernel[grid](
        q, k, cos, sin,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        B, H, L, D,
        BLOCK_HEAD=BLOCK_HEAD, BLOCK_SEQ=BLOCK_SEQ, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps
    )
