import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['BT', 'BK', 'BV']
)
@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(
    q_ptr, k_ptr, v_ptr, h_ptr, g_ptr, do_ptr, dh_ptr,
    dq_ptr, dk_ptr, dg_ptr,
    BT, BK, BV,
    stride_qz, stride_qh, stride_qb, stride_qt, stride_qk,
    stride_kz, stride_kh, stride_kb, stride_kk, stride_kt,
    stride_vz, stride_vh, stride_vb, stride_vk, stride_vt,
    stride_hz, stride_hh, stride_hb, stride_hk, stride_ht,
    stride_gz, stride_gh, stride_gb, stride_gk, stride_gt,
    stride_doz, stride_doh, stride_dob, stride_dot, stride_dok,
    stride_dhz, stride_dhh, stride_dhb, stride_dht, stride_dhk,
    stride_dqz, stride_dqh, stride_dqb, stride_dqt, stride_dqk,
    stride_dkz, stride_dkh, stride_dkb, stride_dkk, stride_dkt,
    stride_dgz, stride_dgh, stride_dgb, stride_dgk, stride_dgt,
    NK, NT, B, H
):
    # Program IDs
    pid_k = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_bh = tl.program_id(2)

    # Compute batch and head indices
    b = pid_bh // H
    h = pid_bh % H

    # Initialize zero tensors for gradients
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BK, BT], dtype=tl.float32)
    b_dg = tl.zeros([BT], dtype=tl.float32)

    # Offsets
    q_offset = b * stride_qb + h * stride_qh
    k_offset = b * stride_kb + h * stride_kh
    v_offset = b * stride_vb + h * stride_vh
    h_offset = b * stride_hb + h * stride_hh
    g_offset = b * stride_gb + h * stride_gh
    do_offset = b * stride_dob + h * stride_doh
    dh_offset = b * stride_dhb + h * stride_dhh

    # Loop over V
    for v in range(0, BV):
        # Boundary checks
        qv = tl.load(q_ptr + q_offset + pid_t * stride_qt + pid_k * stride_qk, mask=pid_t < BT and pid_k < BK, other=0.0)
        kv = tl.load(k_ptr + k_offset + pid_k * stride_kk + v * stride_kt, mask=pid_k < BK and v < BV, other=0.0)
        gv = tl.load(g_ptr + g_offset + pid_t * stride_gt, mask=pid_t < BT, other=0.0)

        # Perform calculations (simplified for example)
        b_dq += qv * kv
        b_dk += kv * gv
        b_dg += gv * qv

    # Store results
    tl.store(dq_ptr + q_offset + pid_t * stride_dqt + pid_k * stride_dqk, b_dq, mask=pid_t < BT and pid_k < BK)
    tl.store(dk_ptr + k_offset + pid_k * stride_dkk + pid_t * stride_dkt, b_dk, mask=pid_k < BK and pid_t < BT)
    tl.store(dg_ptr + g_offset + pid_t * stride_dgt, b_dg, mask=pid_t < BT)

def chunk_bwd_dqkg_fn(q, k, v, h, g, do, dh, BT, BK, BV, B, H):
    # Allocate output tensors
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dg = torch.zeros_like(g)

    # Get strides
    stride_q = q.stride()
    stride_k = k.stride()
    stride_v = v.stride()
    stride_h = h.stride()
    stride_g = g.stride()
    stride_do = do.stride()
    stride_dh = dh.stride()
    stride_dq = dq.stride()
    stride_dk = dk.stride()
    stride_dg = dg.stride()

    # Launch kernel
    grid = (NK, NT, B * H)
    chunk_simple_gla_bwd_kernel_dqkg[grid](
        q, k, v, h, g, do, dh,
        dq, dk, dg,
        BT, BK, BV,
        *stride_q, *stride_k, *stride_v, *stride_h, *stride_g,
        *stride_do, *stride_dh, *stride_dq, *stride_dk, *stride_dg,
        NK, NT, B, H
    )

    return dq, dk, dg
