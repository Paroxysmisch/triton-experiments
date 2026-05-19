import torch
import triton
import triton.language as tl

@triton.jit
def q_kernel_per_block_int8(
    q_ptr, q_int8_ptr, q_scale_ptr,
    M, K,
    stride_qm, stride_qk,
    stride_q8m, stride_q8k,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid // num_pid_m
    pid_k = pid % num_pid_m

    # Compute offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    # Load input block
    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    q_block = tl.load(q_ptrs, mask=mask, other=0.0)
    
    # Compute scale (max absolute value in block)
    q_abs_max = tl.max(tl.abs(q_block), axis=1)
    scale = q_abs_max / 127.0  # Scale to int8 range
    
    # Quantize to int8
    q_int8_block = tl.math.round(q_block / scale[:, None])
    q_int8_block = tl.math.clamp(q_int8_block, -127.0, 127.0)
    
    # Store results
    q8_ptrs = q_int8_ptr + offs_m[:, None] * stride_q8m + offs_k[None, :] * stride_q8k
    scale_ptrs = q_scale_ptr + offs_m
    
    tl.store(q8_ptrs, q_int8_block, mask=mask)
    tl.store(scale_ptrs, scale, mask=offs_m < M)

@triton.jit
def k_kernel_per_block_int8(
    k_ptr, k_int8_ptr, k_scale_ptr,
    N, K,
    stride_kn, stride_kk,
    stride_k8n, stride_k8k,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Similar structure to q_kernel but for key matrix
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid // num_pid_n
    pid_k = pid % num_pid_n

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    k_ptrs = k_ptr + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
    mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    k_block = tl.load(k_ptrs, mask=mask, other=0.0)
    
    k_abs_max = tl.max(tl.abs(k_block), axis=1)
    scale = k_abs_max / 127.0
    
    k_int8_block = tl.math.round(k_block / scale[:, None])
    k_int8_block = tl.math.clamp(k_int8_block, -127.0, 127.0)
    
    k8_ptrs = k_int8_ptr + offs_n[:, None] * stride_k8n + offs_k[None, :] * stride_k8k
    scale_ptrs = k_scale_ptr + offs_n
    
    tl.store(k8_ptrs, k_int8_block, mask=mask)
    tl.store(scale_ptrs, scale, mask=offs_n < N)

def per_block_int8(q, k, BLKQ=32, BLKK=32):
    """
    Wrapper function to handle the int8 quantization process
    """
    M, K = q.shape
    N, K = k.shape
    
    # Prepare output tensors
    q_int8 = torch.empty_like(q, dtype=torch.int8)
    k_int8 = torch.empty_like(k, dtype=torch.int8)
    q_scale = torch.empty((M,), dtype=torch.float32, device=q.device)
    k_scale = torch.empty((N,), dtype=torch.float32, device=k.device)
    
    # Launch kernels
    grid_q = (triton.cdiv(M, BLKQ) * triton.cdiv(K, BLKK),)
    grid_k = (triton.cdiv(N, BLKQ) * triton.cdiv(K, BLKK),)
    
    q_kernel_per_block_int8[grid_q](
        q, q_int8, q_scale,
        M, K,
        q.stride(0), q.stride(1),
        q_int8.stride(0), q_int8.stride(1),
        BLKQ, BLKK
    )
    
    k_kernel_per_block_int8[grid_k](
        k, k_int8, k_scale,
        N, K,
        k.stride(0), k.stride(1),
        k_int8.stride(0), k_int8.stride(1),
        BLKQ, BLKK
    )
    
    return q_int8, q_scale, k_int8, k_scale
