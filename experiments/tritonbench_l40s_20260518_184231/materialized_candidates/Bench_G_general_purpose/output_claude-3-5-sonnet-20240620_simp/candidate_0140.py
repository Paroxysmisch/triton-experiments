import triton
import triton.language as tl
import torch

@triton.jit
def fwd_decay_cumsum_kernel(
    g_ptr, g_o_ptr,
    stride_b, stride_h, stride_t,
    B, H, T,
    scale: tl.float32,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    bid = pid // H
    hid = pid % H

    # Compute pointers
    g_offset = bid * stride_b + hid * stride_h
    g_ptr = g_ptr + g_offset
    g_o_ptr = g_o_ptr + g_offset

    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Compute offsets for this program
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < T

    # Load and compute cumsum with decay
    for t in range(0, T, BLOCK_SIZE):
        curr_mask = mask & (t + offs < T)
        x = tl.load(g_ptr + (t + offs) * stride_t, mask=curr_mask, other=0.0)
        
        # Apply decay and accumulate
        decay = tl.exp(scale * offs)
        acc = tl.where(curr_mask, x * decay, acc)
        
        # Store result
        tl.store(g_o_ptr + (t + offs) * stride_t, acc, mask=curr_mask)
        
        # Update accumulator for next iteration
        if t + BLOCK_SIZE < T:
            acc = acc * tl.exp(scale * BLOCK_SIZE)

@triton.jit
def prepare_qg_kg_kernel(
    q_ptr, k_ptr, g_ptr,
    qg_ptr, kg_ptr,
    stride_b, stride_h, stride_t,
    B, H, T,
    scale: tl.float32,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    bid = pid // H
    hid = pid % H

    # Compute base pointers
    base_offset = bid * stride_b + hid * stride_h
    q_base = q_ptr + base_offset
    k_base = k_ptr + base_offset
    g_base = g_ptr + base_offset
    qg_base = qg_ptr + base_offset
    kg_base = kg_ptr + base_offset

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < T

    for t in range(0, T, BLOCK_SIZE):
        curr_mask = mask & (t + offs < T)
        
        # Load inputs
        q = tl.load(q_base + (t + offs) * stride_t, mask=curr_mask, other=0.0)
        k = tl.load(k_base + (t + offs) * stride_t, mask=curr_mask, other=0.0)
        g = tl.load(g_base + (t + offs) * stride_t, mask=curr_mask, other=0.0)
        
        # Compute exponential scaling
        exp_scale = tl.exp(scale * offs)
        
        # Transform inputs
        qg = q * g * exp_scale
        kg = k * exp_scale
        
        # Store results
        tl.store(qg_base + (t + offs) * stride_t, qg, mask=curr_mask)
        tl.store(kg_base + (t + offs) * stride_t, kg, mask=curr_mask)

@triton.jit
def bwd_decay_global_cumsum_kernel(
    dq_inner_ptr, dq_inter_ptr,
    dk_inner_ptr, dk_inter_ptr,
    q_ptr, k_ptr, g_ptr,
    dg_ptr,
    stride_b, stride_h, stride_t,
    B, H, T,
    scale: tl.float32,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    bid = pid // H
    hid = pid % H

    # Compute base pointers
    base_offset = bid * stride_b + hid * stride_h
    dg_base = dg_ptr + base_offset
    q_base = q_ptr + base_offset
    k_base = k_ptr + base_offset
    g_base = g_ptr + base_offset
    
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < T

    # Initialize accumulator for gradients
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for t in range(0, T, BLOCK_SIZE):
        curr_mask = mask & (t + offs < T)
        
        # Load inputs
        dq_inner = tl.load(dq_inner_ptr + base_offset + (t + offs) * stride_t, mask=curr_mask, other=0.0)
        dq_inter = tl.load(dq_inter_ptr + base_offset + (t + offs) * stride_t, mask=curr_mask, other=0.0)
        dk_inner = tl.load(dk_inner_ptr + base_offset + (t + offs) * stride_t, mask=curr_mask, other=0.0)
        dk_inter = tl.load(dk_inter_ptr + base_offset + (t + offs) * stride_t, mask=curr_mask, other=0.0)
        q = tl.load(q_base + (t + offs) * stride_t, mask=curr_mask, other=0.0)
        k = tl.load(k_base + (t + offs) * stride_t, mask=curr_mask, other=0.0)
        g = tl.load(g_base + (t + offs) * stride_t, mask=curr_mask, other=0.0)

        # Compute gradient contributions
        exp_scale = tl.exp(scale * offs)
        dg = (dq_inner * q + dk_inner * k) * exp_scale + (dq_inter + dk_inter) * g
        acc = acc + dg

        # Store accumulated gradients
        tl.store(dg_base + (t + offs) * stride_t, acc, mask=curr_mask)

def launch_fwd_decay_cumsum(g, scale, BLOCK_SIZE=128):
    B, H, T = g.shape
    g_o = torch.empty_like(g)
    
    def grid(meta):
        return (B * H,)

    fwd_decay_cumsum_kernel[grid](
        g, g_o,
        g.stride(0), g.stride(1), g.stride(2),
        B, H, T, scale,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return g_o

def launch_prepare_qg_kg(q, k, g, scale, BLOCK_SIZE=128):
    B, H, T = q.shape
    qg = torch.empty_like(q)
    kg = torch.empty_like(k)
    
    def grid(meta):
        return (B * H,)

    prepare_qg_kg_kernel[grid](
        q, k, g,
        qg, kg,
        q.stride(0), q.stride(1), q.stride(2),
        B, H, T, scale,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return qg, kg

def launch_bwd_decay_global_cumsum(dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, scale, BLOCK_SIZE=128):
    B, H, T = q.shape
    dg = torch.empty_like(g)
    
    def grid(meta):
        return (B * H,)

    bwd_decay_global_cumsum_kernel[grid](
        dq_inner, dq_inter,
        dk_inner, dk_inter,
        q, k, g,
        dg,
        q.stride(0), q.stride(1), q.stride(2),
        B, H, T, scale,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return dg
