import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BT': 64, 'BK': 64, 'BV': 64}, num_warps=4),
        triton.Config({'BT': 64, 'BK': 32, 'BV': 32}, num_warps=8),
    ],
    key=['BT', 'BK', 'BV']
)
@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(
    # Input tensors
    q, k, v, h, g, do, dh,
    # Output gradients
    dq, dk, dg,
    # Strides for q, k, v, h, g, do, dh
    s_qh, s_qt, s_qd,
    s_kh, s_kt, s_kd,
    s_vh, s_vt, s_vd,
    s_gh, s_gt, s_gd,
    s_oh, s_ot, s_od,
    s_hh, s_ht, s_hd,
    # Tensor dimensions and parameters
    B, H, T, scale,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr,
    USE_GATE: tl.constexpr,
    CHECK_BOUNDS: tl.constexpr
):
    # Program IDs determine the block's responsibility
    i_k = tl.program_id(0)  # Index along NK dimension (key chunks)
    i_t = tl.program_id(1)  # Index along NT dimension (time chunks)
    i_bh = tl.program_id(2)  # Combined batch and head index
    
    # Offsets and initializations
    i_h = i_bh % H
    o_i = tl.arange(0, BT)
    p_q = tl.make_block_ptr(
        base=q + i_bh * s_qh,
        shape=(T, DK),
        strides=(s_qt, s_qd),
        offsets=(i_t * BT, i_k * BK),
        block_shape=(BT, BK),
        order=(1, 0)
    )
    # Similar block pointers for k, v, g, do, dh, dq, dk, dg

    # Initialize gradient accumulators
    acc_dq = tl.zeros((BT, BK), dtype=tl.float32)
    acc_dk = tl.zeros((BT, BK), dtype=tl.float32)
    acc_dg = tl.zeros((BT,), dtype=tl.float32)  # Assuming g is per position

    # Loop over value dimension chunks
    for i_v in range(tl.cdiv(DV, BV)):
        # Load current blocks of v, do, dh
        p_v = tl.make_block_ptr(...)
        p_do = tl.make_block_ptr(...)
        p_dh = tl.make_block_ptr(...)
        
        b_v = tl.load(p_v, boundary_check=(0, 1) if CHECK_BOUNDS else ())
        b_do = tl.load(p_do, boundary_check=(0, 1) if CHECK_BOUNDS else ())
        b_dh = tl.load(p_dh, boundary_check=(0, 1) if CHECK_BOUNDS else ())

        # Load g block if used
        if USE_GATE:
            p_g = tl.make_block_ptr(...)
            b_g = tl.load(p_g, boundary_check=(0, 1) if CHECK_BOUNDS else ())
            gate_scale = tl.exp(b_g * scale)  # Example g usage

        # Compute attention gradient components
        b_s = tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
        b_s = b_s * gate_scale if USE_GATE else b_s  # Apply gating effect

        # Compute dq component
        dq_part = tl.dot(b_s, tl.load(p_k), allow_tf32=False)
        acc_dq += dq_part

        # Compute dk component
        dk_part = tl.dot(tl.trans(b_s), tl.load(p_q), allow_tf32=False)
        acc_dk += dk_part

        # Compute dg component (example derivative calculation)
        if USE_GATE:
            dg_part = tl.sum(b_s * b_g * scale, axis=1)  # Chain rule through exp
            acc_dg += dg_part

    # Store accumulated gradients
    p_dq = tl.make_block_ptr(...)
    tl.store(p_dq, acc_dq.to(p_dq.dtype.element_ty))
    p_dk = tl.make_block_ptr(...)
    tl.store(p_dk, acc_dk.to(p_dk.dtype.element_ty))
    if USE_GATE:
        p_dg = tl.make_block_ptr(...)
        tl.store(p_dg, acc_dg.to(p_dg.dtype.element_ty))

def chunk_bwd_dqkg_fn(q, k, v, h, g, do, dh):
    B, H, T, DK = q.shape
    _, _, _, DV = v.shape

    # Determine kernel configurations
    BT = 64
    BK = min(triton.next_power_of_2(DK), 64)
    BV = min(triton.next_power_of_2(DV), 64)
    NK, NT = triton.cdiv(DK, BK), triton.cdiv(T, BT)
    
    # Initialize output gradients
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dg = torch.empty_like(g) if g is not None else None

    # Define grid and launch kernel
    grid = (NK, NT, B * H)
    chunk_simple_gla_bwd_kernel_dqkg[grid](
        q, k, v, h, g, do, dh, dq, dk, dg,
        # Stride parameters and other constants...
        B=B, H=H, T=T, scale=DK**-0.5,
        BT=BT, BK=BK, BV=BV, DK=DK, DV=DV,
        USE_GATE=g is not None,
        CHECK_BOUNDS=True  # If T not divisible by BT
    )
    return dq, dk, dg
