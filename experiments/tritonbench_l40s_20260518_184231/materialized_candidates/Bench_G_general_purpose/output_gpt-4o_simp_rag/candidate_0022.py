import torch
import triton
import triton.language as tl

@triton.jit
def q_kernel_per_block_int8(
    q_ptr, q_int8_ptr, q_scale_ptr,
    M, N, 
    stride_qm, stride_qn,
    stride_q_int8m, stride_q_int8n,
    stride_q_scale,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qn
    q_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    q_max = tl.max(q, axis=1)
    q_min = tl.min(q, axis=1)
    q_scale = (q_max - q_min) / 255.0
    q_zero_point = tl.cast(q_min / q_scale, dtype=tl.int8)
    
    q_quantized = tl.cast((q / q_scale[:, None]) + q_zero_point[:, None], dtype=tl.int8)
    
    q_int8_ptrs = q_int8_ptr + offs_m[:, None] * stride_q_int8m + offs_n[None, :] * stride_q_int8n
    q_scale_ptrs = q_scale_ptr + offs_m * stride_q_scale
    
    tl.store(q_int8_ptrs, q_quantized, mask=q_mask)
    tl.store(q_scale_ptrs, q_scale)

@triton.jit
def k_kernel_per_block_int8(
    k_ptr, k_int8_ptr, k_scale_ptr,
    M, N, 
    stride_km, stride_kn,
    stride_k_int8m, stride_k_int8n,
    stride_k_scale,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    k_ptrs = k_ptr + offs_m[:, None] * stride_km + offs_n[None, :] * stride_kn
    k_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    k = tl.load(k_ptrs, mask=k_mask, other=0.0)
    
    k_max = tl.max(k, axis=1)
    k_min = tl.min(k, axis=1)
    k_scale = (k_max - k_min) / 255.0
    k_zero_point = tl.cast(k_min / k_scale, dtype=tl.int8)
    
    k_quantized = tl.cast((k / k_scale[:, None]) + k_zero_point[:, None], dtype=tl.int8)
    
    k_int8_ptrs = k_int8_ptr + offs_m[:, None] * stride_k_int8m + offs_n[None, :] * stride_k_int8n
    k_scale_ptrs = k_scale_ptr + offs_m * stride_k_scale
    
    tl.store(k_int8_ptrs, k_quantized, mask=k_mask)
    tl.store(k_scale_ptrs, k_scale)

def per_block_int8(q, k, BLKQ, BLKK):
    Mq, Nq = q.shape
    Mk, Nk = k.shape
    
    q_int8 = torch.empty((Mq, Nq), dtype=torch.int8, device=q.device)
    q_scale = torch.empty((Mq,), dtype=torch.float32, device=q.device)
    
    k_int8 = torch.empty((Mk, Nk), dtype=torch.int8, device=k.device)
    k_scale = torch.empty((Mk,), dtype=torch.float32, device=k.device)
    
    grid_q = lambda META: (triton.cdiv(Mq, META['BLOCK_SIZE_M']) * triton.cdiv(Nq, META['BLOCK_SIZE_N']),)
    grid_k = lambda META: (triton.cdiv(Mk, META['BLOCK_SIZE_M']) * triton.cdiv(Nk, META['BLOCK_SIZE_N']),)
    
    q_kernel_per_block_int8[grid_q](
        q, q_int8, q_scale,
        Mq, Nq,
        q.stride(0), q.stride(1),
        q_int8.stride(0), q_int8.stride(1),
        q_scale.stride(0),
        BLOCK_SIZE_M=BLKQ, BLOCK_SIZE_N=BLKQ
    )
    
    k_kernel_per_block_int8[grid_k](
        k, k_int8, k_scale,
        Mk, Nk,
        k.stride(0), k.stride(1),
        k_int8.stride(0), k_int8.stride(1),
        k_scale.stride(0),
        BLOCK_SIZE_M=BLKK, BLOCK_SIZE_N=BLKK
    )
    
    return q_int8, q_scale, k_int8, k_scale
