import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8),
    ],
    key=['BT', 'BK', 'BV'],
)
@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, h_ptr, g_ptr,
    do_ptr, dh_ptr,
    dq_ptr, dk_ptr, dg_ptr,
    # Matrix dimensions
    BT, BK, BV,
    # Strides
    stride_qb, stride_qh, stride_qt, stride_qk,
    stride_kb, stride_kh, stride_kt, stride_kv,
    stride_vb, stride_vh, stride_vt, stride_vv,
    stride_hb, stride_hh, stride_ht, stride_hv,
    stride_gb, stride_gh, stride_gt, stride_gk,
    # Other strides for gradients
    stride_dob, stride_doh, stride_dot, stride_dov,
    stride_dhb, stride_dhh, stride_dht, stride_dhv,
    stride_dqb, stride_dqh, stride_dqt, stride_dqk,
    stride_dkb, stride_dkh, stride_dkt, stride_dkv,
    stride_dgb, stride_dgh, stride_dgt, stride_dgk,
    # Number of heads and batch size
    H, B,
    NT, NK, NV,
):
    # Program ID
    pid_k = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_bh = tl.program_id(2)
    
    # Calculate batch and head indices
    b_idx = pid_bh // H
    h_idx = pid_bh % H

    # Initialize accumulators for gradients
    b_dq = tl.zeros([BK], dtype=tl.float32)
    b_dk = tl.zeros([BK], dtype=tl.float32)
    b_dg = tl.zeros([BK], dtype=tl.float32)
    
    # Offsets for the current block
    k_offs = pid_k * BK + tl.arange(0, BK)
    t_offs = pid_t * BT + tl.arange(0, BT)
    
    # Mask for bounds checking
    k_mask = k_offs < NK
    t_mask = t_offs < NT

    # Load query, key, and gate for current position
    q = tl.load(q_ptr + b_idx * stride_qb + h_idx * stride_qh + t_offs[:, None] * stride_qt + k_offs[None, :] * stride_qk,
                mask=t_mask[:, None] & k_mask[None, :])
    k = tl.load(k_ptr + b_idx * stride_kb + h_idx * stride_kh + t_offs[:, None] * stride_kt + k_offs[None, :] * stride_kv,
                mask=t_mask[:, None] & k_mask[None, :])
    g = tl.load(g_ptr + b_idx * stride_gb + h_idx * stride_gh + t_offs[:, None] * stride_gt + k_offs[None, :] * stride_gk,
                mask=t_mask[:, None] & k_mask[None, :])

    # Load output gradients
    do = tl.load(do_ptr + b_idx * stride_dob + h_idx * stride_doh + t_offs * stride_dot,
                 mask=t_mask)
    dh = tl.load(dh_ptr + b_idx * stride_dhb + h_idx * stride_dhh + t_offs * stride_dht,
                 mask=t_mask)

    # Compute attention scores
    scores = tl.dot(q, k.transpose())
    scores = scores * (1.0 / tl.sqrt(BK))
    
    # Apply gating mechanism
    gated_scores = scores * tl.sigmoid(g)
    
    # Compute gradients
    d_gated = do[:, None] * dh[None, :]
    d_scores = d_gated * tl.sigmoid(g)
    d_g = d_gated * scores * tl.sigmoid(g) * (1.0 - tl.sigmoid(g))
    
    # Accumulate gradients
    b_dq += tl.dot(d_scores, k)
    b_dk += tl.dot(d_scores.transpose(), q)
    b_dg += tl.sum(d_g, axis=0)

    # Store results
    tl.store(dq_ptr + b_idx * stride_dqb + h_idx * stride_dqh + t_offs * stride_dqt + k_offs * stride_dqk,
             b_dq, mask=k_mask)
    tl.store(dk_ptr + b_idx * stride_dkb + h_idx * stride_dkh + t_offs * stride_dkt + k_offs * stride_dkv,
             b_dk, mask=k_mask)
    tl.store(dg_ptr + b_idx * stride_dgb + h_idx * stride_dgh + t_offs * stride_dgt + k_offs * stride_dgk,
             b_dg, mask=k_mask)

def chunk_bwd_dqkg_fn(q, k, v, h, g, do, dh):
    """
    Wrapper function for the backward pass of gated attention.
    
    Args:
        q: Query tensor of shape (B, H, T, K)
        k: Key tensor of shape (B, H, T, K)
        v: Value tensor of shape (B, H, T, V)
        h: Hidden states tensor of shape (B, H, T, V)
        g: Gating tensor of shape (B, H, T, K)
        do: Output gradient tensor of shape (B, H, T, V)
        dh: Hidden gradient tensor of shape (B, H, T, V)
    
    Returns:
        Tuple of gradients (dq, dk, dg)
    """
    batch_size, num_heads, seq_len, dim_k = q.shape
    _, _, _, dim_v = v.shape

    # Initialize output tensors
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dg = torch.zeros_like(g)

    # Configure grid
    grid = (
        triton.cdiv(dim_k, 128),  # NK
        triton.cdiv(seq_len, 128),  # NT
        batch_size * num_heads,    # B*H
    )

    # Launch kernel
    chunk_simple_gla_bwd_kernel_dqkg[grid](
        q, k, v, h, g, do, dh,
        dq, dk, dg,
        seq_len, dim_k, dim_v,
        *get_strides(q, k, v, h, g, do, dh, dq, dk, dg),
        num_heads, batch_size,
        seq_len, dim_k, dim_v,
    )

    return dq, dk, dg

def get_strides(*tensors):
    """Helper function to get strides of all tensors"""
    strides = []
    for tensor in tensors:
        strides.extend([tensor.stride(0), tensor.stride(1), tensor.stride(2), tensor.stride(3)])
    return strides
