import triton
import triton.language as tl

inv_ln2 = 1.44269504  # Predefined constant for numerical stability

# Forward kernel: Cumulative sum with decay
@triton.jit
def fwd_decay_cumsum(
    g_ptr, g_o_ptr,
    s_qk_h, s_qk_t, s_qk_d,  # Strides for head, time, depth dimensions
    B, H, T, DK,  # Tensor dimensions
    scale: tl.constexpr,  # Scaling factor
    BT: tl.constexpr, BK: tl.constexpr  # Tile sizes
):
    # Parallelize over K, T, and BH dimensions
    i_k = tl.program_id(0)
    i_c = tl.program_id(1)
    i_bh = tl.program_id(2)
    
    # Compute initial pointers for the current block
    g_start = i_bh * s_qk_h + i_c * BT * s_qk_t + i_k * BK
    g_o_start = i_bh * s_qk_h + i_c * BT * s_qk_t + i_k * BK
    
    cum_decay = tl.zeros([BK], dtype=tl.float32)
    
    for i in range(BT):
        t_idx = i_c * BT + i
        # Check if current time step is within tensor bounds
        mask_t = t_idx < T
        offsets = g_start + i * s_qk_t + tl.arange(0, BK)
        mask = mask_t & (offsets < (i_bh + 1) * s_qk_h + T * s_qk_t + DK)
        
        # Load current g value and update cumulative sum
        curr_g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
        cum_decay += curr_g * inv_ln2
        
        # Store result and increment pointers
        tl.store(g_o_ptr + offsets, cum_decay.to(g_o_ptr.dtype.element_ty), mask=mask)

# Preparation kernel: Compute qg and kg with exponential decay
@triton.jit
def prepare_qg_kg(
    q_ptr, k_ptr, g_ptr, qg_ptr, kg_ptr,
    s_qk_h, s_qk_t, s_qk_d,
    B, H, T, DK,
    scale: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr
):
    i_k = tl.program_id(0)
    i_c = tl.program_id(1)
    i_bh = tl.program_id(2)
    
    # Get last decay value in the current block
    last_t_idx = i_c * BT + BT - 1
    mask_last_t = last_t_idx < T
    last_g_offset = i_bh * s_qk_h + last_t_idx * s_qk_t + i_k * BK
    last_g = tl.load(g_ptr + last_g_offset + tl.arange(0, BK), 
                    mask=mask_last_t & (last_g_offset < (i_bh + 1)*s_qk_h + T*s_qk_t + DK), 
                    other=0.0)
    
    # Process each element in the block
    for i in range(BT):
        t_idx = i_c * BT + i
        mask_t = t_idx < T
        offsets = i_bh * s_qk_h + t_idx * s_qk_t + i_k * BK
        
        # Load data with boundary checks
        mask = mask_t & (offsets < (i_bh + 1)*s_qk_h + T*s_qk_t + DK)
        q = tl.load(q_ptr + offsets, mask=mask, other=0.0)
        k = tl.load(k_ptr + offsets, mask=mask, other=0.0)
        g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
        
        # Compute transformed values
        q_transformed = q * tl.math.exp2(g * inv_ln2) * scale
        k_transformed = k * tl.math.exp2((last_g - g) * inv_ln2)
        
        # Store results
        tl.store(qg_ptr + offsets, q_transformed.to(qg_ptr.dtype.element_ty), mask=mask)
        tl.store(kg_ptr + offsets, k_transformed.to(kg_ptr.dtype.element_ty), mask=mask)

# Backward kernel: Gradient computation for decay
@triton.jit
def bwd_decay_global_cumsum(
    dq_in_ptr, dq_out_ptr, dk_in_ptr, dk_out_ptr,
    q_ptr, k_ptr, g_ptr, dg_ptr,
    s_qk_h, s_qk_t, s_qk_d,
    B, H, T, DK,
    scale: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr
):
    i_k = tl.program_id(0)
    i_c = tl.program_id(1)
    i_bh = tl.program_id(2)
    
    cum_grad = tl.zeros([BK], dtype=tl.float32)
    
    # Process elements in reverse order
    for i in range(BT-1, -1, -1):
        t_idx = i_c * BT + i
        mask_t = t_idx < T
        offsets = i_bh * s_qk_h + t_idx * s_qk_t + i_k * BK
        mask = mask_t & (offsets < (i_bh + 1)*s_qk_h + T*s_qk_t + DK)
        
        # Load necessary data
        dq = tl.load(dq_in_ptr + offsets, mask=mask, other=0.0)
        dk = tl.load(dk_in_ptr + offsets, mask=mask, other=0.0)
        q = tl.load(q_ptr + offsets, mask=mask, other=0.0)
        k = tl.load(k_ptr + offsets, mask=mask, other=0.0)
        g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
        
        # Compute gradients
        dg_current = (dq * q - dk * k) * inv_ln2
        cum_grad += dg_current
        tl.store(dg_ptr + offsets, cum_grad.to(dg_ptr.dtype.element_ty), mask=mask)
        
        # Update output gradients
        tl.store(dq_out_ptr + offsets, dq.to(dq_out_ptr.dtype.element_ty), mask=mask)
        tl.store(dk_out_ptr + offsets, dk.to(dk_out_ptr.dtype.element_ty), mask=mask)

# Wrapper functions for kernel launches
def fwd_decay_cumsum_launch(g, g_o, scale):
    B, H, T, DK = g.shape
    grid = (
        triton.cdiv(DK, 64),  # K dimension tiles
        triton.cdiv(T, 64),    # T dimension tiles
        B * H                 # Batch*Heads
    )
    fwd_decay_cumsum[grid](
        g, g_o,
        g.stride(1), g.stride(2), g.stride(3),
        B, H, T, DK, scale,
        BT=64, BK=64
    )

def prepare_qg_kg_launch(q, k, g, qg, kg, scale):
    B, H, T, DK = q.shape
    grid = (
        triton.cdiv(DK, 64),
        triton.cdiv(T, 64),
        B * H
    )
    prepare_qg_kg[grid](
        q, k, g, qg, kg,
        q.stride(1), q.stride(2), q.stride(3),
        B, H, T, DK, scale,
        BT=64, BK=64
    )

def bwd_decay_global_cumsum_launch(dq_in, dq_out, dk_in, dk_out, q, k, g, dg, scale):
    B, H, T, DK = q.shape
    grid = (
        triton.cdiv(DK, 64),
        triton.cdiv(T, 64),
        B * H
    )
    bwd_decay_global_cumsum[grid](
        dq_in, dq_out, dk_in, dk_out,
        q, k, g, dg,
        q.stride(1), q.stride(2), q.stride(3),
        B, H, T, DK, scale,
        BT=64, BK=64
    )
