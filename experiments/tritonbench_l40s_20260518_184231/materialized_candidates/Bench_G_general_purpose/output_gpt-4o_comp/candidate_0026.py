import torch
import triton
import triton.language as tl

@triton.jit
def q_kernel_per_block_int8(q_ptr, q_int8_ptr, q_scale_ptr, BLKQ, num_cols, stride, **meta):
    pid = tl.program_id(axis=0)
    block_start = pid * BLKQ

    offsets = block_start + tl.arange(0, BLKQ)
    mask = offsets < num_cols

    q_block = tl.load(q_ptr + offsets, mask=mask)
    
    max_abs_val = tl.max(tl.abs(q_block), axis=0)
    scale = 127.0 / max_abs_val
    q_block_scaled = q_block * scale
    q_block_int8 = tl.libdevice.rint(q_block_scaled).to(tl.int8)
    
    tl.store(q_int8_ptr + offsets, q_block_int8, mask=mask)
    tl.store(q_scale_ptr + pid, scale)

@triton.jit
def k_kernel_per_block_int8(k_ptr, k_int8_ptr, k_scale_ptr, BLKK, num_cols, stride, **meta):
    pid = tl.program_id(axis=0)
    block_start = pid * BLKK

    offsets = block_start + tl.arange(0, BLKK)
    mask = offsets < num_cols

    k_block = tl.load(k_ptr + offsets, mask=mask)
    
    max_abs_val = tl.max(tl.abs(k_block), axis=0)
    scale = 127.0 / max_abs_val
    k_block_scaled = k_block * scale
    k_block_int8 = tl.libdevice.rint(k_block_scaled).to(tl.int8)
    
    tl.store(k_int8_ptr + offsets, k_block_int8, mask=mask)
    tl.store(k_scale_ptr + pid, scale)

def per_block_int8(q, k, BLKQ, BLKK):
    assert q.ndim == 2 and k.ndim == 2, "Input tensors must be 2D"
    
    num_rows_q, num_cols_q = q.shape
    num_rows_k, num_cols_k = k.shape
    
    assert num_cols_q == num_cols_k, "Query and key matrices must have the same number of columns"
    
    q_int8 = torch.empty((num_rows_q, num_cols_q), dtype=torch.int8, device=q.device)
    k_int8 = torch.empty((num_rows_k, num_cols_k), dtype=torch.int8, device=k.device)
    
    q_scale = torch.empty((num_rows_q // BLKQ,), dtype=torch.float32, device=q.device)
    k_scale = torch.empty((num_rows_k // BLKK,), dtype=torch.float32, device=k.device)
    
    grid_q = (num_cols_q + BLKQ - 1) // BLKQ
    grid_k = (num_cols_k + BLKK - 1) // BLKK
    
    q_kernel_per_block_int8[grid_q](q, q_int8, q_scale, BLKQ, num_cols_q, q.stride(0))
    k_kernel_per_block_int8[grid_k](k, k_int8, k_scale, BLKK, num_cols_k, k.stride(0))
    
    return q_int8, k_int8, q_scale, k_scale

# Example usage
q = torch.randn(1024, 256, device='cuda')
k = torch.randn(1024, 256, device='cuda')
BLKQ = 128
BLKK = 128

q_int8, k_int8, q_scale, k_scale = per_block_int8(q, k, BLKQ, BLKK)
