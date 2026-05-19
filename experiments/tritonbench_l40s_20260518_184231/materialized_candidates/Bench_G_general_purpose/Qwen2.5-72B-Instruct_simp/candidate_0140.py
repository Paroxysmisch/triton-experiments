import triton
import triton.language as tl

# 1. fwd_decay_cumsum
@triton.jit
def fwd_decay_cumsum(g_ptr, g_o_ptr, decay_ptr, B, H, T, BT, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BT
    decay = tl.load(decay_ptr)
    
    for i in range(block_start, min(block_start + BT, T)):
        offset = tl.arange(0, BLOCK_SIZE)
        g_idx = block_start + offset
        g = tl.load(g_ptr + g_idx, mask=g_idx < T, other=0.0)
        
        if i == block_start:
            g_o = g
        else:
            g_o = g + decay * g_o
        
        tl.store(g_o_ptr + g_idx, g_o, mask=g_idx < T)

# 2. prepare_qg_kg
@triton.jit
def prepare_qg_kg(q_ptr, k_ptr, g_ptr, qg_ptr, kg_ptr, scale_ptr, B, H, T, BT, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BT
    scale = tl.load(scale_ptr)
    
    for i in range(block_start, min(block_start + BT, T)):
        offset = tl.arange(0, BLOCK_SIZE)
        q_idx = block_start + offset
        k_idx = block_start + offset
        g_idx = block_start + offset
        
        q = tl.load(q_ptr + q_idx, mask=q_idx < T, other=0.0)
        k = tl.load(k_ptr + k_idx, mask=k_idx < T, other=0.0)
        g = tl.load(g_ptr + g_idx, mask=g_idx < T, other=0.0)
        
        qg = q * tl.exp(g * scale)
        kg = k * tl.exp(g * scale)
        
        tl.store(qg_ptr + q_idx, qg, mask=q_idx < T)
        tl.store(kg_ptr + k_idx, kg, mask=k_idx < T)

# 3. bwd_decay_global_cumsum
@triton.jit
def bwd_decay_global_cumsum(dq_inner_ptr, dq_inter_ptr, dk_inner_ptr, dk_inter_ptr, q_ptr, k_ptr, g_ptr, dg_ptr, decay_ptr, B, H, T, BT, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BT
    decay = tl.load(decay_ptr)
    
    for i in range(block_start, min(block_start + BT, T)):
        offset = tl.arange(0, BLOCK_SIZE)
        q_idx = block_start + offset
        k_idx = block_start + offset
        g_idx = block_start + offset
        
        dq_inner = tl.load(dq_inner_ptr + q_idx, mask=q_idx < T, other=0.0)
        dq_inter = tl.load(dq_inter_ptr + q_idx, mask=q_idx < T, other=0.0)
        dk_inner = tl.load(dk_inner_ptr + k_idx, mask=k_idx < T, other=0.0)
        dk_inter = tl.load(dk_inter_ptr + k_idx, mask=k_idx < T, other=0.0)
        q = tl.load(q_ptr + q_idx, mask=q_idx < T, other=0.0)
        k = tl.load(k_ptr + k_idx, mask=k_idx < T, other=0.0)
        g = tl.load(g_ptr + g_idx, mask=g_idx < T, other=0.0)
        
        dg = (dq_inner + dq_inter) * q * decay + (dk_inner + dk_inter) * k * decay
        
        tl.store(dg_ptr + g_idx, dg, mask=g_idx < T)

import torch

def launch_fwd_decay_cumsum(g, g_o, decay, B, H, T, BT, BLOCK_SIZE):
    grid = (T // BT, B * H)
    fwd_decay_cumsum[grid](g, g_o, decay, B, H, T, BT, BLOCK_SIZE)

def launch_prepare_qg_kg(q, k, g, qg, kg, scale, B, H, T, BT, BLOCK_SIZE):
    grid = (T // BT, B * H)
    prepare_qg_kg[grid](q, k, g, qg, kg, scale, B, H, T, BT, BLOCK_SIZE)

def launch_bwd_decay_global_cumsum(dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, dg, decay, B, H, T, BT, BLOCK_SIZE):
    grid = (T // BT, B * H)
    bwd_decay_global_cumsum[grid](dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, dg, decay, B, H, T, BT, BLOCK_SIZE)

# Example usage with PyTorch tensors
B, H, T = 2, 4, 128
BT, BLOCK_SIZE = 32, 128

# Input tensors
g = torch.randn((B, H, T), device='cuda')
q = torch.randn((B, H, T), device='cuda')
k = torch.randn((B, H, T), device='cuda')
dq_inner = torch.randn((B, H, T), device='cuda')
dq_inter = torch.randn((B, H, T), device='cuda')
dk_inner = torch.randn((B, H, T), device='cuda')
dk_inter = torch.randn((B, H, T), device='cuda')

# Output tensors
g_o = torch.zeros((B, H, T), device='cuda')
qg = torch.zeros((B, H, T), device='cuda')
kg = torch.zeros((B, H, T), device='cuda')
dg = torch.zeros((B, H, T), device='cuda')

# Constants
decay = torch.tensor(0.9, device='cuda')
scale = torch.tensor(0.5, device='cuda')

# Launch kernels
launch_fwd_decay_cumsum(g, g_o, decay, B, H, T, BT, BLOCK_SIZE)
launch_prepare_qg_kg(q, k, g, qg, kg, scale, B, H, T, BT, BLOCK_SIZE)
launch_bwd_decay_global_cumsum(dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, dg, decay, B, H, T, BT, BLOCK_SIZE)
