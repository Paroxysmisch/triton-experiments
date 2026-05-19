import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    USE_FP8: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    num_pid_n = tl.cdiv(N_CTX, BLOCK_N)
    num_pid_h = H
    num_pid_z = Z
    
    # Initialize offsets
    pid_m = pid // (num_pid_n * num_pid_h * num_pid_z)
    pid_n = (pid % (num_pid_n * num_pid_h * num_pid_z)) // (num_pid_h * num_pid_z)
    pid_h = (pid % (num_pid_h * num_pid_z)) // num_pid_z
    pid_z = pid % num_pid_z

    # Initialize pointers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    
    # Initialize Q, K, V pointers
    q_ptrs = Q + (pid_z * stride_qz + pid_h * stride_qh + 
                  offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    k_ptrs = K + (pid_z * stride_kz + pid_h * stride_kh + 
                  offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
    v_ptrs = V + (pid_z * stride_vz + pid_h * stride_vh + 
                  offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)

    # Load Q, K, V
    if USE_FP8:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0).to(tl.float8e4m3)
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0).to(tl.float8e4m3)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0).to(tl.float8e4m3)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)

    # Compute attention scores
    s = tl.dot(q, tl.trans(k))
    s = s * sm_scale

    # Apply causal mask if needed
    if IS_CAUSAL:
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        s = s * causal_mask + float("-inf") * ~causal_mask

    # Apply softmax
    s = s - tl.max(s, 1)[:, None]
    s = tl.exp(s)
    s = s / tl.sum(s, 1)[:, None]

    # Compute output
    o = tl.dot(s, v)

    # Store output
    out_ptrs = Out + (pid_z * stride_oz + pid_h * stride_oh + 
                      offs_m[:, None] * stride_om + offs_k[None, :] * stride_on)
    tl.store(out_ptrs, o, mask=offs_m[:, None] < N_CTX)

def triton_fa(q, k, v, sm_scale, is_causal=False):
    """
    Wrapper function for the forward pass of Flash Attention
    
    Arguments:
        q: Query tensor of shape (Z, H, M, K)
        k: Key tensor of shape (Z, H, N, K)
        v: Value tensor of shape (Z, H, N, K)
        sm_scale: Scaling factor for attention scores
        is_causal: Whether to apply causal masking
    
    Returns:
        out: Output tensor of shape (Z, H, M, K)
    """
    # Extract dimensions
    Z, H, M, K = q.shape
    _, _, N, _ = k.shape
    
    # Determine block sizes based on K dimension
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = K
    
    # Initialize output
    out = torch.empty_like(q)
    
    # Compute grid size
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N) * H * Z,)
    
    # Determine whether to use FP8
    USE_FP8 = K <= 64  # Use FP8 for smaller dimensions
    
    # Launch kernel
    _fwd_kernel[grid](
        q, k, v, sm_scale, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        USE_FP8=USE_FP8,
        IS_CAUSAL=is_causal,
        num_warps=4,
        num_stages=2,
    )
    
    return out
