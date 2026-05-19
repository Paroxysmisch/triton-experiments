import triton
import triton.language as tl
import torch
import math

# Constants for block sizes
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32
THETA = 10000.0  # Base for rotary embeddings

@triton.jit
def rms_matmul_rbe_kernel(
    # Pointers to matrices
    x_ptr, w_ptr, rms_w_ptr, output_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    # Optional parameters
    apply_rotary: tl.constexpr,
    position_offset: tl.constexpr,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Computes RMS-normalized matrix multiplication with optional rotary embeddings
    """
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Block pointers
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Pointers to current blocks
    x_block_ptr = x_ptr + block_start_m * stride_xm
    w_block_ptr = w_ptr + block_start_n * stride_wn
    
    # RMS Normalization
    rms_sum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Compute RMS norm
    for k in range(0, K, BLOCK_SIZE_K):
        x_mask = (block_start_m + tl.arange(0, BLOCK_SIZE_M)) < M
        k_mask = tl.arange(0, BLOCK_SIZE_K) + k < K
        
        x = tl.load(x_block_ptr + k * stride_xk, mask=x_mask[:, None] & k_mask[None, :])
        rms_sum += tl.sum(x * x, axis=1)
    
    # Compute RMS scale
    rms_scale = 1.0 / tl.sqrt(rms_sum / K + 1e-6)
    rms_w = tl.load(rms_w_ptr + tl.arange(0, BLOCK_SIZE_M))
    rms_scale = rms_scale * rms_w
    
    # Main matmul loop with normalized inputs
    for k in range(0, K, BLOCK_SIZE_K):
        x_mask = (block_start_m + tl.arange(0, BLOCK_SIZE_M)) < M
        k_mask = tl.arange(0, BLOCK_SIZE_K) + k < K
        n_mask = (block_start_n + tl.arange(0, BLOCK_SIZE_N)) < N
        
        # Load x and normalize
        x = tl.load(x_block_ptr + k * stride_xk, mask=x_mask[:, None] & k_mask[None, :])
        x = x * rms_scale[:, None]
        
        # Apply rotary embeddings if needed
        if apply_rotary:
            position = position_offset + block_start_m + tl.arange(0, BLOCK_SIZE_M)
            theta = tl.arange(0, head_size) // 2 * 2
            theta = 1.0 / (THETA ** (theta / head_size))
            
            freq = position[:, None] * theta[None, :]
            cos = tl.cos(freq)
            sin = tl.sin(freq)
            
            x_rot = tl.where(
                tl.arange(0, BLOCK_SIZE_K)[None, :] % 2 == 0,
                x * cos - tl.roll(x, 1, 1) * sin,
                x * cos + tl.roll(x, -1, 1) * sin
            )
            x = x_rot
        
        w = tl.load(w_block_ptr + k * stride_wk, mask=k_mask[:, None] & n_mask[None, :])
        
        acc += tl.dot(x, w)
    
    # Store output
    output_mask = (
        (block_start_m + tl.arange(0, BLOCK_SIZE_M))[:, None] < M
    ) & (
        (block_start_n + tl.arange(0, BLOCK_SIZE_N))[None, :] < N
    )
    output_ptr = output_ptr + block_start_m * stride_om + block_start_n * stride_on
    tl.store(output_ptr, acc, mask=output_mask)

def rms_matmul_rbe_wrapper(x, w, rms_w, apply_rotary=False, position_offset=0):
    """
    Wrapper function for the RMS matrix multiplication kernel with rotary embeddings
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, hidden_dim)
        w: Weight tensor of shape (hidden_dim, output_dim)
        rms_w: RMS weight tensor of shape (hidden_dim,)
        apply_rotary: Whether to apply rotary embeddings
        position_offset: Starting position for rotary embeddings
    """
    batch_size, seq_len, hidden_dim = x.shape
    output_dim = w.shape[1]
    
    # Ensure contiguous inputs
    x = x.contiguous()
    w = w.contiguous()
    rms_w = rms_w.contiguous()
    
    # Create output tensor
    output = torch.empty((batch_size, seq_len, output_dim), 
                        device=x.device, dtype=x.dtype)
    
    # Calculate grid and block sizes
    grid = lambda META: (
        triton.cdiv(seq_len, META['BLOCK_SIZE_M']) *
        triton.cdiv(output_dim, META['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    rms_matmul_rbe_kernel[grid](
        x_ptr=x, w_ptr=w, rms_w_ptr=rms_w, output_ptr=output,
        M=seq_len, N=output_dim, K=hidden_dim,
        stride_xm=hidden_dim, stride_xk=1,
        stride_wk=output_dim, stride_wn=1,
        stride_om=output_dim, stride_on=1,
        apply_rotary=apply_rotary,
        position_offset=position_offset,
        num_heads=hidden_dim // 64,  # Assuming 64-dimensional heads
        head_size=64,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output
