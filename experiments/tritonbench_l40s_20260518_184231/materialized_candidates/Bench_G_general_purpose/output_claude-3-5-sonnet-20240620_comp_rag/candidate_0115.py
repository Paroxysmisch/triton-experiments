import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=8),
    ],
    key=['BT', 'BK', 'BV']
)
@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(
    # Pointers to tensors
    q_ptr, k_ptr, v_ptr, h_ptr, g_ptr,
    do_ptr, dh_ptr, dq_ptr, dk_ptr, dg_ptr,
    # Dimensions and strides
    B, H, T, D,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    # Constants
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    # Program ID
    pid_k = tl.program_id(0)  # NK
    pid_t = tl.program_id(1)  # NT
    pid_bh = tl.program_id(2)  # B*H
    
    # Initialize offsets
    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_t = pid_t * BT + tl.arange(0, BT)
    
    # Batch/head index calculations
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    # Initialize gradient accumulators
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dg = tl.zeros([BT, BK], dtype=tl.float32)
    
    # Load block pointers
    q_block_ptr = tl.make_block_ptr(
        q_ptr + b_idx * stride_qb + h_idx * stride_qh,
        (T, D),
        (stride_qt, stride_qd),
        (offs_t, offs_k),
        (BT, BK),
        (1, 0)
    )
    
    k_block_ptr = tl.make_block_ptr(
        k_ptr + b_idx * stride_kb + h_idx * stride_kh,
        (T, D),
        (stride_kt, stride_kd),
        (offs_t, offs_k),
        (BT, BK),
        (1, 0)
    )
    
    # Main computation loop
    for v in range(0, tl.cdiv(T, BV)):
        # Load query, key, value blocks
        b_q = tl.load(q_block_ptr, boundary_check=(0, 1))
        b_k = tl.load(k_block_ptr, boundary_check=(0, 1))
        
        # Load output gradients
        offs_v = v * BV + tl.arange(0, BV)
        b_do = tl.load(do_ptr + b_idx * stride_vb + h_idx * stride_vh + 
                      offs_t[:, None] * stride_vt + offs_v[None, :] * stride_vd,
                      boundary_check=(0, 1))
        
        # Compute attention scores
        b_s = tl.dot(b_q, tl.trans(b_k))
        b_s = tl.where(offs_t[:, None] >= offs_t[None, :], b_s, 0)
        
        # Scale and apply softmax
        b_s = b_s * (1.0 / tl.sqrt(float(D)))
        b_s = tl.softmax(b_s, axis=-1)
        
        # Compute gradients
        b_dq += tl.dot(b_do, tl.trans(b_k))
        b_dk += tl.dot(tl.trans(b_do), b_q)
        b_dg += tl.dot(b_do, b_k) * tl.sigmoid(b_s)
        
        # Advance block pointers
        q_block_ptr = tl.advance(q_block_ptr, (BT, 0))
        k_block_ptr = tl.advance(k_block_ptr, (BT, 0))
    
    # Store results
    dq_ptr = dq_ptr + b_idx * stride_qb + h_idx * stride_qh + offs_t[:, None] * stride_qt + offs_k[None, :] * stride_qd
    dk_ptr = dk_ptr + b_idx * stride_kb + h_idx * stride_kh + offs_t[:, None] * stride_kt + offs_k[None, :] * stride_kd
    dg_ptr = dg_ptr + b_idx * stride_qb + h_idx * stride_qh + offs_t[:, None] * stride_qt + offs_k[None, :] * stride_qd
    
    tl.store(dq_ptr, b_dq, boundary_check=(0, 1))
    tl.store(dk_ptr, b_dk, boundary_check=(0, 1))
    tl.store(dg_ptr, b_dg, boundary_check=(0, 1))

def chunk_bwd_dqkg_fn(q, k, v, h, g, do, dh):
    # Extract dimensions
    B, H, T, D = q.shape
    
    # Initialize output tensors
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dg = torch.empty_like(g)
    
    # Calculate grid dimensions
    BK = min(triton.next_power_of_2(D), 64)
    BT = min(triton.next_power_of_2(T), 64)
    BV = min(triton.next_power_of_2(D), 64)
    
    grid = (triton.cdiv(D, BK), triton.cdiv(T, BT), B * H)
    
    # Launch kernel
    chunk_simple_gla_bwd_kernel_dqkg[grid](
        q, k, v, h, g, do, dh, dq, dk, dg,
        B, H, T, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        BT=BT, BK=BK, BV=BV,
    )
    
    return dq, dk, dg
