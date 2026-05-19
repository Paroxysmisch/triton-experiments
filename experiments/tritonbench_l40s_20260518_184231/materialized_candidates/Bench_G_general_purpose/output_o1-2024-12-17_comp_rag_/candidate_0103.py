import triton
import triton.language as tl
import torch

@triton.jit
def _rotary_kernel(
    Q_ptr, K_ptr, Cos_ptr, Sin_ptr,
    stride_qH, stride_qS, stride_qD,
    stride_kH, stride_kS, stride_kD,
    stride_cosH, stride_cosS, stride_cosD,
    stride_sinH, stride_sinS, stride_sinD,
    H, S, D,
    BLOCK_HEAD: tl.constexpr, BLOCK_SEQ: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    h_block = tl.program_id(0)
    s_block = tl.program_id(1)

    h_range = h_block * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    s_range = s_block * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    d_range = tl.arange(0, BLOCK_DMODEL)

    h_mask = h_range < H
    s_mask = s_range < S
    d_mask = d_range < D

    # Create mesh to load data
    H_indices = h_range[:, None, None]   # (BLOCK_HEAD, 1, 1)
    S_indices = s_range[None, :, None]  # (1, BLOCK_SEQ, 1)
    D_indices = d_range[None, None, :]  # (1, 1, BLOCK_DMODEL)

    # Masks
    head_mask = h_mask[:, None, None]
    seq_mask = s_mask[None, :, None]
    dim_mask = d_mask[None, None, :]
    mask = head_mask & seq_mask & dim_mask

    # Offsets for Q
    q_offset = (H_indices * stride_qH +
                S_indices * stride_qS +
                D_indices * stride_qD)
    # Offsets for K
    k_offset = (H_indices * stride_kH +
                S_indices * stride_kS +
                D_indices * stride_kD)
    # Offsets for Cos, Sin
    cos_offset = (H_indices * stride_cosH +
                  S_indices * stride_cosS +
                  D_indices * stride_cosD)
    sin_offset = (H_indices * stride_sinH +
                  S_indices * stride_sinS +
                  D_indices * stride_sinD)

    # Load
    q = tl.where(mask, tl.load(Q_ptr + q_offset, mask=mask), 0.0)
    k = tl.where(mask, tl.load(K_ptr + k_offset, mask=mask), 0.0)
    cos_val = tl.where(mask, tl.load(Cos_ptr + cos_offset, mask=mask), 0.0)
    sin_val = tl.where(mask, tl.load(Sin_ptr + sin_offset, mask=mask), 0.0)

    # Split q, k into two halves
    half_dim = BLOCK_DMODEL // 2
    q0 = q[..., :half_dim]
    q1 = q[..., half_dim:]
    k0 = k[..., :half_dim]
    k1 = k[..., half_dim:]

    cos0 = cos_val[..., :half_dim]
    cos1 = cos_val[..., half_dim:]
    sin0 = sin_val[..., :half_dim]
    sin1 = sin_val[..., half_dim:]

    # Rotary transform
    out_q0 = q0 * cos0 - q1 * sin0
    out_q1 = q0 * sin1 + q1 * cos1
    out_k0 = k0 * cos0 - k1 * sin0
    out_k1 = k0 * sin1 + k1 * cos1

    # Recombine
    out_q = tl.concatenate([out_q0, out_q1], axis=-1)
    out_k = tl.concatenate([out_k0, out_k1], axis=-1)

    # Store back
    tl.store(Q_ptr + q_offset, out_q, mask=mask)
    tl.store(K_ptr + k_offset, out_k, mask=mask)

def rotary_emb_fwd(Q, K, Cos, Sin):
    B, H, S, D = Q.shape
    assert Q.shape == K.shape, "Q and K must have the same shape"
    assert Cos.shape == Sin.shape, "Cos and Sin must have the same shape"
    assert Cos.shape == (B, H, S, D), "Cos, Sin shapes must match Q, K"

    BLOCK_HEAD = 1
    BLOCK_SEQ = 1
    BLOCK_DMODEL = D

    grid = (H, S)
    num_warps = 4 if D >= 64 else 1

    triton.run(
        _rotary_kernel,
        grid=grid,
        num_warps=num_warps,
        args=[
            Q, K, Cos, Sin,
            Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(1), K.stride(2), K.stride(3),
            Cos.stride(1), Cos.stride(2), Cos.stride(3),
            Sin.stride(1), Sin.stride(2), Sin.stride(3),
            H, S, D
        ],
        constants=[BLOCK_HEAD, BLOCK_SEQ, BLOCK_DMODEL]
    )
    return Q, K
