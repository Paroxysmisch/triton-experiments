import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(
    # Pointers to input tensors
    q_ptr, k_ptr, v_ptr,
    h_ptr, g_ptr,
    do_ptr, dh_ptr,
    # Pointers to output tensors (gradients)
    dq_ptr, dk_ptr, dg_ptr,
    # Strides / shapes for q, k, v, h, g, do, dh
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_hb, stride_hh, stride_ht, stride_hd,
    stride_gb, stride_gh, stride_gt, stride_gd,
    stride_dob, stride_doh, stride_dot, stride_dod,
    stride_dhb, stride_dhh, stride_dht, stride_dhd,
    # Strides for dq, dk, dg
    stride_dqb, stride_dqh, stride_dqt, stride_dqd,
    stride_dkb, stride_dkh, stride_dkt, stride_dkd,
    stride_dgb, stride_dgh, stride_dgt, stride_dgd,
    # B: batch size, H: number of heads, T: sequence length, D: head dimension
    B, H, T, D,
    # Block sizes
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr
):
    """
    Kernel that computes the backward gradients for Q (dq), K (dk), and
    a gating mechanism (dg) in attention-like modules. It processes
    the corresponding input tensors and accumulates partial gradients
    over blocks, storing the final results in dq, dk, dg.
    """

    # Program IDs to identify the block in the 3D grid:
    #   i_tv ranges over BV blocks (or whichever dimension we assign for 'V')
    #   i_tk ranges over BK blocks (or whichever dimension we assign for 'K')
    #   i_bh identifies batch * head blocks
    i_tv = tl.program_id(0)
    i_tk = tl.program_id(1)
    i_bh = tl.program_id(2)

    # Recover the batch index and head index from i_bh
    b_idx = i_bh // H
    h_idx = i_bh % H

    # Create a range for the block of size BT along the time dimension
    # We'll map row_i for the T-dimension processing within each block
    row_i = tl.arange(0, BT)

    # Offsets to load data from pointers
    # Typically, each tensor is shaped like (B, H, T, D) and we manage
    # offsets using the provided strides
    q_offset  = b_idx * stride_qb  + h_idx * stride_qh
    k_offset  = b_idx * stride_kb  + h_idx * stride_kh
    v_offset  = b_idx * stride_vb  + h_idx * stride_vh
    h_offset  = b_idx * stride_hb  + h_idx * stride_hh
    g_offset  = b_idx * stride_gb  + h_idx * stride_gh
    do_offset = b_idx * stride_dob + h_idx * stride_d
