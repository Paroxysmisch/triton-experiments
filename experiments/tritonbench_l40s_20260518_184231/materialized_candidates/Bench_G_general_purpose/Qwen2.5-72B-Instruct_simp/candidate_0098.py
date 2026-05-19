import triton
import triton.language as tl

@triton.jit
def _rotary_kernel(
    Q, K, Cos, Sin,
    stride_qh, stride_qd, stride_qm,
    stride_kh, stride_kd, stride_km,
    stride_ch, stride_cd, stride_cm,
    stride_sh, stride_sd, stride_sm,
    max_total_len, HEAD_Q, HEAD_K,
    BLOCK_HEAD: tl.constexpr, BLOCK_SEQ: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    # Get the current block's coordinates in the 2D grid
    pid_head = tl.program_id(axis=0)
    pid_seq = tl.program_id(axis=1)

    # Compute the starting indices for the current block
    head_start = pid_head * BLOCK_HEAD
    seq_start = pid_seq * BLOCK_SEQ

    # Initialize the offsets for the current block
    offsets_head = head_start + tl.arange(0, BLOCK_HEAD)
    offsets_seq = seq_start + tl.arange(0, BLOCK_SEQ)
    offsets_dmodel = tl.arange(0, BLOCK_DMODEL)

    # Create masks to handle boundary conditions
    mask_head = offsets_head < HEAD_Q
    mask_seq = offsets_seq < max_total_len

    # Load the slices from Q and K
    Q_offsets = (offsets_head[:, None, None] * stride_qh + offsets_seq[None, :, None] * stride_qm + offsets_dmodel[None, None, :] * stride_qd)
    K_offsets = (offsets_head[:, None, None] * stride_kh + offsets_seq[None, :, None] * stride_km + offsets_dmodel[None, None, :] * stride_kd)
    Q_block = tl.load(Q + Q_offsets, mask=mask_head[:, None, None] & mask_seq[None, :, None], other=0.0)
    K_block = tl.load(K + K_offsets, mask=mask_head[:, None, None] & mask_seq[None, :, None], other=0.0)

    # Load the cosine and sine embeddings
    Cos_offsets = (offsets_head[:, None, None] * stride_ch + offsets_seq[None, :, None] * stride_cm + offsets_dmodel[None, None, :] * stride_cd)
    Sin_offsets = (offsets_head[:, None, None] * stride_sh + offsets_seq[None, :, None] * stride_sm + offsets_dmodel[None, None, :] * stride_sd)
    Cos_block = tl.load(Cos + Cos_offsets, mask=mask_head[:, None, None] & mask_seq[None, :, None], other=0.0)
    Sin_block = tl.load(Sin + Sin_offsets, mask=mask_head[:, None, None] & mask_seq[None, :, None], other=0.0)

    # Apply the rotary transformation
    Q_rotated = Q_block * Cos_block - tl.flip(Q_block, 2) * Sin_block
    K_rotated = K_block * Cos_block - tl.flip(K_block, 2) * Sin_block

    # Store the results back into Q and K
    tl.store(Q + Q_offsets, Q_rotated, mask=mask_head[:, None, None] & mask_seq[None, :, None])
    tl.store(K + K_offsets, K_rotated, mask=mask_head[:, None, None] & mask_seq[None, :, None])

import torch

def rotary_emb_fwd(Q, K, Cos, Sin, max_total_len, HEAD_Q, HEAD_K, BLOCK_HEAD, BLOCK_SEQ, BLOCK_DMODEL):
    # Determine the grid size for execution
    grid = (HEAD_Q // BLOCK_HEAD, max_total_len // BLOCK_SEQ)

    # Determine the number of warps based on the dimension size
    num_warps = 4 if BLOCK_DMODEL <= 256 else 8

    # Launch the kernel
    _rotary_kernel[grid](
        Q, K, Cos, Sin,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        Cos.stride(0), Cos.stride(1), Cos.stride(2),
        Sin.stride(0), Sin.stride(1), Sin.stride(2),
        max_total_len, HEAD_Q, HEAD_K,
        BLOCK_HEAD, BLOCK_SEQ, BLOCK_DMODEL,
        num_warps=num_warps
    )

# Example usage
Q = torch.randn((8, 128, 64), device='cuda')
K = torch.randn((8, 128, 64), device='cuda')
Cos = torch.randn((8, 128, 64), device='cuda')
Sin = torch.randn((8, 128, 64), device='cuda')
max_total_len = 128
HEAD_Q = 8
HEAD_K = 8
BLOCK_HEAD = 1
BLOCK_SEQ = 32
BLOCK_DMODEL = 64

rotary_emb_fwd(Q, K, Cos, Sin, max_total_len, HEAD_Q, HEAD_K, BLOCK_HEAD, BLOCK_SEQ, BLOCK_DMODEL)
