import triton
import triton.language as tl
import torch

# Kernel for forward decay cumulative sum
@triton.jit
def fwd_decay_cumsum_kernel(
    g_ptr, g_o_ptr, decay, inv_ln2, 
    T, DK, BT,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_start = pid * BT
    cum_decay = tl.zeros([DK], dtype=tl.float32)
    
    for t in range(0, BT):
        row_idx = row_start + t
        mask = row_idx < T
        g = tl.load(g_ptr + row_idx * DK + tl.arange(0, DK), mask=mask)
        g_scaled = g * inv_ln2
        cum_decay += g_scaled
        tl.store(g_o_ptr + row_idx * DK + tl.arange(0, DK), cum_decay, mask=mask)

# Kernel to prepare qg and kg
@triton.jit
def prepare_qg_kg_kernel(
    q_ptr, k_ptr, g_ptr, qg_ptr, kg_ptr, 
    T, DK, BT,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_start = pid * BT
    
    for t in range(0, BT):
        row_idx = row_start + t
        mask = row_idx < T
        q = tl.load(q_ptr + row_idx * DK + tl.arange(0, DK), mask=mask)
        k = tl.load(k_ptr + row_idx * DK + tl.arange(0, DK), mask=mask)
        g = tl.load(g_ptr + row_idx * DK + tl.arange(0, DK), mask=mask)
        
        exp_decay = tl.exp(-g)
        qg = q * exp_decay
        kg = k * exp_decay
        
        tl.store(qg_ptr + row_idx * DK + tl.arange(0, DK), qg, mask=mask)
        tl.store(kg_ptr + row_idx * DK + tl.arange(0, DK), kg, mask=mask)

# Kernel for backward decay cumulative sum
@triton.jit
def bwd_decay_global_cumsum_kernel(
    dq_inner_ptr, dq_inter_ptr, dk_inner_ptr, dk_inter_ptr, 
    q_ptr, k_ptr, g_ptr, dg_ptr,
    T, DK, BT,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_start = pid * BT
    cum_decay_grad = tl.zeros([DK], dtype=tl.float32)
    
    for t in range(BT-1, -1, -1):
        row_idx = row_start + t
        mask = row_idx < T
        
        dq_inner = tl.load(dq_inner_ptr + row_idx * DK + tl.arange(0, DK), mask=mask)
        dq_inter = tl.load(dq_inter_ptr + row_idx * DK + tl.arange(0, DK), mask=mask)
        dk_inner = tl.load(dk_inner_ptr + row_idx * DK + tl.arange(0, DK), mask=mask)
        dk_inter = tl.load(dk_inter_ptr + row_idx * DK + tl.arange(0, DK), mask=mask)
        
        q = tl.load(q_ptr + row_idx * DK + tl.arange(0, DK), mask=mask)
        k = tl.load(k_ptr + row_idx * DK + tl.arange(0, DK), mask=mask)
        g = tl.load(g_ptr + row_idx * DK + tl.arange(0, DK), mask=mask)
        
        grad_decay = (dq_inner + dq_inter) * q + (dk_inner + dk_inter) * k
        cum_decay_grad += grad_decay
        tl.store(dg_ptr + row_idx * DK + tl.arange(0, DK), cum_decay_grad, mask=mask)

# Launch functions
def launch_fwd_decay_cumsum(g, g_o, decay, inv_ln2, BT):
    T, DK = g.shape
    grid = (T // BT,)
    fwd_decay_cumsum_kernel[grid](g, g_o, decay, inv_ln2, T, DK, BT, BLOCK_SIZE=128)

def launch_prepare_qg_kg(q, k, g, qg, kg, BT):
    T, DK = q.shape
    grid = (T // BT,)
    prepare_qg_kg_kernel[grid](q, k, g, qg, kg, T, DK, BT, BLOCK_SIZE=128)

def launch_bwd_decay_global_cumsum(dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, dg, BT):
    T, DK = q.shape
    grid = (T // BT,)
    bwd_decay_global_cumsum_kernel[grid](dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, dg, T, DK, BT, BLOCK_SIZE=128)

# Example usage
# Initialize tensors and call launch functions as needed.
