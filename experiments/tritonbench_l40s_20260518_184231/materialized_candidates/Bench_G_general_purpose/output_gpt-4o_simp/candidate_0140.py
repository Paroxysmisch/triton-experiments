import triton
import triton.language as tl

# 1. Forward Decay Cumulative Sum Kernel
@triton.jit
def fwd_decay_cumsum_kernel(g_ptr, g_o_ptr, decay, B, H, T, BT, **meta):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    
    # Calculate offsets
    offset = b_idx * H * T + h_idx * T
    g = g_ptr + offset
    g_o = g_o_ptr + offset
    
    # Initialize cumulative sum
    cumsum = tl.zeros((BT,), dtype=tl.float32)
    
    for t in range(0, T, BT):
        g_tile = tl.load(g + t + tl.arange(0, BT))
        cumsum = decay * cumsum + g_tile
        tl.store(g_o + t + tl.arange(0, BT), cumsum)

# 2. Prepare QG and KG Kernel
@triton.jit
def prepare_qg_kg_kernel(q_ptr, k_ptr, g_ptr, qg_ptr, kg_ptr, scale, B, H, T, BT, **meta):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    
    # Calculate offsets
    offset = b_idx * H * T + h_idx * T
    q = q_ptr + offset
    k = k_ptr + offset
    g = g_ptr + offset
    qg = qg_ptr + offset
    kg = kg_ptr + offset
    
    for t in range(0, T, BT):
        q_tile = tl.load(q + t + tl.arange(0, BT))
        k_tile = tl.load(k + t + tl.arange(0, BT))
        g_tile = tl.load(g + t + tl.arange(0, BT))
        
        qg_tile = q_tile * tl.exp(scale * g_tile)
        kg_tile = k_tile * tl.exp(scale * g_tile)
        
        tl.store(qg + t + tl.arange(0, BT), qg_tile)
        tl.store(kg + t + tl.arange(0, BT), kg_tile)

# 3. Backward Decay Global Cumulative Sum Kernel
@triton.jit
def bwd_decay_global_cumsum_kernel(dq_inner_ptr, dq_inter_ptr, dk_inner_ptr, dk_inter_ptr, q_ptr, k_ptr, g_ptr, dg_ptr, B, H, T, BT, **meta):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    
    # Calculate offsets
    offset = b_idx * H * T + h_idx * T
    dq_inner = dq_inner_ptr + offset
    dq_inter = dq_inter_ptr + offset
    dk_inner = dk_inner_ptr + offset
    dk_inter = dk_inter_ptr + offset
    q = q_ptr + offset
    k = k_ptr + offset
    g = g_ptr + offset
    dg = dg_ptr + offset
    
    for t in range(0, T, BT):
        dq_inner_tile = tl.load(dq_inner + t + tl.arange(0, BT))
        dq_inter_tile = tl.load(dq_inter + t + tl.arange(0, BT))
        dk_inner_tile = tl.load(dk_inner + t + tl.arange(0, BT))
        dk_inter_tile = tl.load(dk_inter + t + tl.arange(0, BT))
        
        # Compute gradient contributions
        grad_contrib = dq_inner_tile + dq_inter_tile + dk_inner_tile + dk_inter_tile
        
        # Update dg
        dg_tile = tl.load(dg + t + tl.arange(0, BT))
        dg_tile += grad_contrib
        tl.store(dg + t + tl.arange(0, BT), dg_tile)


def launch_fwd_decay_cumsum(g, g_o, decay, B, H, T, BT):
    grid = (B, H)
    fwd_decay_cumsum_kernel[grid](g, g_o, decay, B, H, T, BT)

def launch_prepare_qg_kg(q, k, g, qg, kg, scale, B, H, T, BT):
    grid = (B, H)
    prepare_qg_kg_kernel[grid](q, k, g, qg, kg, scale, B, H, T, BT)

def launch_bwd_decay_global_cumsum(dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, dg, B, H, T, BT):
    grid = (B, H)
    bwd_decay_global_cumsum_kernel[grid](dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, dg, B, H, T, BT)
