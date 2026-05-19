import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"num_warps": 4}, num_stages=1),
        triton.Config({"num_warps": 8}, num_stages=1),
    ],
    key=['BT', 'BK', 'BV']
)
@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(
    q, k, v, h, g, do, dh,
    dq, dk, dg,
    B, H, T, D,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr
):
    # Program IDs
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b = i_bh // H
    i_h = i_bh % H

    # Initialize zero tensors for gradients
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BK, BT], dtype=tl.float32)
    b_dg = tl.zeros([BT, BK], dtype=tl.float32)

    # Create pointers to input tensors
    p_q = tl.make_block_ptr(q + i_bh * T * D, (T, D), (1, T), (0, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * T * D, (D, T), (D, 1), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * T * D, (T, D), (1, T), (0, i_v * BV), (BT, BV), (1, 0))
    p_h = tl.make_block_ptr(h + i_bh * T * D, (T, D), (1, T), (0, i_v * BV), (BT, BV), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * T * D, (T, D), (1, T), (0, i_v * BV), (BT, BV), (1, 0))
    p_do = tl.make_block_ptr(do + i_bh * T * D, (T, D), (1, T), (0, i_v * BV), (BT, BV), (1, 0))
    p_dh = tl.make_block_ptr(dh + i_bh * T * D, (T, D), (1, T), (0, i_v * BV), (BT, BV), (1, 0))

    NT = tl.cdiv(T, BT)
    for i in range(0, NT):
        # Load blocks
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))

        # Compute gradients
        b_dq += tl.dot(b_do, b_k.T)
        b_dk += tl.dot(b_q.T, b_do)
        b_dg += b_do * b_h * b_g

        # Advance pointers
        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_h = tl.advance(p_h, (BT, 0))
        p_g = tl.advance(p_g, (BT, 0))
        p_do = tl.advance(p_do, (BT, 0))
        p_dh = tl.advance(p_dh, (BT, 0))

    # Store results
    p_dq = tl.make_block_ptr(dq + i_bh * T * D, (T, D), (1, T), (0, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * T * D, (D, T), (D, 1), (i_k * BK, 0), (BK, BT), (0, 1))
    p_dg = tl.make_block_ptr(dg + i_bh * T * D, (T, D), (1, T), (0, i_v * BV), (BT, BV), (1, 0))

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


def chunk_bwd_dqkg_fn(q, k, v, h, g, do, dh, B, H, T, D):
    # Initialize output tensors
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dg = torch.zeros_like(g)

    # Determine block sizes
    BT = 64
    BK = min(triton.next_power_of_2(D), 64)
    BV = min(triton.next_power_of_2(D), 64)
    NK = triton.cdiv(D, BK)
    NV = triton.cdiv(D, BV)

    # Define grid
    grid = (NV, NK, B * H)

    # Launch the kernel
    chunk_simple_gla_bwd_kernel_dqkg[grid](
        q, k, v, h, g, do, dh,
        dq, dk, dg,
        B, H, T, D,
        BT=BT, BK=BK, BV=BV,
        num_warps=4,
        num_stages=1
    )

    return dq, dk, dg
