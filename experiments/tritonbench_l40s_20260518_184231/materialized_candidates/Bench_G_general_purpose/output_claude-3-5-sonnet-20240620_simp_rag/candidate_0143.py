import triton
import triton.language as tl
import torch

# Constants
BLOCK_SIZE = 128
inv_ln2 = 1.44269504

@triton.jit
def fwd_decay_cumsum_kernel(
    g_ptr, g_out_ptr,
    stride_bh, stride_t, stride_d,
    B, H, T, scale,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr, DIM_K: tl.constexpr
):
    pid_k, pid_c, pid_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    # Calculate input/output pointers
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    g_ptr = g_ptr + pid_bh * stride_bh + pid_c * BLOCK_T * DIM_K + offs_k
    g_out_ptr = g_out_ptr + pid_bh * stride_bh + pid_c * BLOCK_T * DIM_K + offs_k
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    mask = offs_k < DIM_K
    
    # Main loop
    for i in range(BLOCK_T):
        g = tl.load(g_ptr, mask=mask, other=0.0).to(tl.float32)
        acc += g * inv_ln2
        tl.store(g_out_ptr, acc, mask=mask)
        g_ptr += DIM_K
        g_out_ptr += DIM_K

@triton.jit
def prepare_qg_kg_kernel(
    q_ptr, k_ptr, g_ptr, qg_ptr, kg_ptr,
    stride_bh, stride_t, stride_d,
    B, H, T, scale,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr, DIM_K: tl.constexpr
):
    pid_k, pid_c, pid_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    # Calculate pointers
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    base_ptr = pid_bh * stride_bh + pid_c * BLOCK_T * DIM_K + offs_k
    
    q_ptr = q_ptr + base_ptr
    k_ptr = k_ptr + base_ptr
    g_ptr = g_ptr + base_ptr
    qg_ptr = qg_ptr + base_ptr
    kg_ptr = kg_ptr + base_ptr
    
    mask = offs_k < DIM_K
    
    # Load last decay value
    last_g = tl.load(g_ptr + (BLOCK_T - 1) * DIM_K, mask=mask, other=0.0)
    
    # Process blocks
    for i in range(BLOCK_T):
        q = tl.load(q_ptr, mask=mask, other=0.0)
        k = tl.load(k_ptr, mask=mask, other=0.0)
        g = tl.load(g_ptr, mask=mask, other=0.0).to(tl.float32)
        
        # Apply transformations
        qg = q * tl.exp2(g) * scale
        kg = k * tl.exp2(last_g - g)
        
        # Store results
        tl.store(qg_ptr, qg, mask=mask)
        tl.store(kg_ptr, kg, mask=mask)
        
        # Advance pointers
        q_ptr += DIM_K
        k_ptr += DIM_K
        g_ptr += DIM_K
        qg_ptr += DIM_K
        kg_ptr += DIM_K

@triton.jit
def bwd_decay_global_cumsum_kernel(
    dq_inner_ptr, dq_inter_ptr, dk_inner_ptr, dk_inter_ptr,
    q_ptr, k_ptr, g_ptr, dg_ptr,
    stride_bh, stride_t, stride_d,
    B, H, T, scale,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr, DIM_K: tl.constexpr
):
    pid_k, pid_c, pid_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    # Calculate base pointers
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    base_ptr = pid_bh * stride_bh + (pid_c * BLOCK_T + BLOCK_T - 1) * DIM_K + offs_k
    
    # Initialize pointers
    ptrs = {
        'q': q_ptr + base_ptr,
        'k': k_ptr + base_ptr,
        'g': g_ptr + base_ptr,
        'dg': dg_ptr + base_ptr,
        'dq_inner': dq_inner_ptr + base_ptr,
        'dk_inner': dk_inner_ptr + base_ptr,
        'dq_inter': dq_inter_ptr + base_ptr,
        'dk_inter': dk_inter_ptr + base_ptr
    }
    
    mask = offs_k < DIM_K
    acc_dg = tl.zeros([BLOCK_K], dtype=tl.float32)
    
    # Backward pass
    for t in range(BLOCK_T-1, -1, -1):
        g = tl.load(ptrs['g'], mask=mask, other=0.0).to(tl.float32)
        
        # Load and process gradients
        dq1 = tl.load(ptrs['dq_inner'], mask=mask, other=0.0)
        dq2 = tl.load(ptrs['dq_inter'], mask=mask, other=0.0)
        dk1 = tl.load(ptrs['dk_inner'], mask=mask, other=0.0)
        dk2 = tl.load(ptrs['dk_inter'], mask=mask, other=0.0)
        
        # Transform gradients
        dq2 = dq2 * tl.exp2(g)
        dk2 = dk2 * tl.exp2(g if t == BLOCK_T-1 else last_g - g)
        
        # Combine gradients
        dq = dq1 + dq2
        dk = dk1 + dk2
        
        # Store intermediate results
        tl.store(ptrs['dq_inter'], dq, mask=mask)
        tl.store(ptrs['dk_inter'], dk, mask=mask)
        
        # Calculate gradient for g
        q = tl.load(ptrs['q'], mask=mask, other=0.0)
        k = tl.load(ptrs['k'], mask=mask, other=0.0)
        dg = dq * q - dk * k
        acc_dg += dg
        
        # Store accumulated gradient
        tl.store(ptrs['dg'], acc_dg, mask=mask)
        
        # Update last_g for next iteration
        if t == BLOCK_T-1:
            last_g = g
            
        # Move pointers
        for ptr in ptrs.values():
            ptr -= DIM_K

def launch_fwd_decay_cumsum(g, B, H, T, scale):
    BLOCK_T = 32
    BLOCK_K = 32
    DIM_K = g.shape[-1]
    
    g_out = torch.empty_like(g)
    grid = (triton.cdiv(DIM_K, BLOCK_K), T // BLOCK_T, B * H)
    
    fwd_decay_cumsum_kernel[grid](
        g, g_out,
        g.stride(0), g.stride(1), g.stride(2),
        B, H, T, scale,
        BLOCK_T, BLOCK_K, DIM_K
    )
    return g_out

def launch_prepare_qg_kg(q, k, g, B, H, T, scale):
    BLOCK_T = 32
    BLOCK_K = 32
    DIM_K = q.shape[-1]
    
    qg = torch.empty_like(q)
    kg = torch.empty_like(k)
    grid = (triton.cdiv(DIM_K, BLOCK_K), T // BLOCK_T, B * H)
    
    prepare_qg_kg_kernel[grid](
        q, k, g, qg, kg,
        q.stride(0), q.stride(1), q.stride(2),
        B, H, T, scale,
        BLOCK_T, BLOCK_K, DIM_K
    )
    return qg, kg

def launch_bwd_decay_global_cumsum(dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, B, H, T, scale):
    BLOCK_T = 32
    BLOCK_K = 32
    DIM_K = q.shape[-1]
    
    dg = torch.empty_like(g)
    grid = (triton.cdiv(DIM_K, BLOCK_K), T // BLOCK_T, B * H)
    
    bwd_decay_global_cumsum_kernel[grid](
        dq_inner, dq_inter, dk_inner, dk_inter,
        q, k, g, dg,
        q.stride(0), q.stride(1), q.stride(2),
        B, H, T, scale,
        BLOCK_T, BLOCK_K, DIM_K
    )
    return dg
