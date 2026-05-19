import torch
import triton
import triton.language as tl
import math

@triton.jit
def rms_matmul_rbe_kernel(
    # Pointers to matrices
    x_ptr, weight_ptr, output_ptr, rms_norm_ptr,
    # Matrix dimensions
    batch_size, seq_len, hidden_dim, head_dim,
    # Strides for tensors
    stride_b, stride_s, stride_h,
    # Additional parameters
    scale, num_warps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Block dimensions
    num_pid_m = tl.cdiv(batch_size * seq_len, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(hidden_dim, BLOCK_SIZE_N)
    
    # Current block
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Initialize offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Load RMS norm values
    b_idx = offs_m // seq_len
    s_idx = offs_m % seq_len
    rms_offs = b_idx * stride_b + s_idx * stride_s
    rms = tl.load(rms_norm_ptr + rms_offs, mask=offs_m < batch_size * seq_len)
    
    # Main loop
    for k in range(0, hidden_dim, BLOCK_SIZE_K):
        # Load x and weight blocks
        x_block_ptr = x_ptr + (offs_m[:, None] * stride_h + (k + offs_k[None, :]))
        w_block_ptr = weight_ptr + ((k + offs_k[:, None]) * hidden_dim + offs_n[None, :])
        
        x = tl.load(x_block_ptr, mask=(offs_m[:, None] < batch_size * seq_len) & 
                                     (k + offs_k[None, :] < hidden_dim))
        w = tl.load(w_block_ptr, mask=(k + offs_k[:, None] < hidden_dim) & 
                                     (offs_n[None, :] < hidden_dim))
        
        # Apply RMS normalization
        x = x * rms[:, None]
        
        # Matrix multiplication
        acc += tl.dot(x, w)
    
    # Apply scale
    acc = acc * scale
    
    # Store output
    out_ptr = output_ptr + offs_m[:, None] * hidden_dim + offs_n[None, :]
    tl.store(out_ptr, acc, mask=(offs_m[:, None] < batch_size * seq_len) & 
                               (offs_n[None, :] < hidden_dim))

def rms_matmul_rbe_wrapper(x, weight, rms_norm, scale=1.0):
    """
    Wrapper function for RMS MatMul with Rotary Embeddings
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, hidden_dim)
        weight: Weight matrix of shape (hidden_dim, hidden_dim)
        rms_norm: RMS normalization values of shape (batch_size, seq_len)
        scale: Scaling factor for the output
    
    Returns:
        Output tensor of shape (batch_size, seq_len, hidden_dim)
    """
    batch_size, seq_len, hidden_dim = x.shape
    assert weight.shape == (hidden_dim, hidden_dim)
    assert rms_norm.shape == (batch_size, seq_len)
    
    # Ensure contiguous tensors
    x = x.contiguous()
    weight = weight.contiguous()
    rms_norm = rms_norm.contiguous()
    
    # Prepare output tensor
    output = torch.empty_like(x)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Calculate grid size
    grid = (triton.cdiv(batch_size * seq_len, BLOCK_SIZE_M) * 
            triton.cdiv(hidden_dim, BLOCK_SIZE_N),)
    
    # Launch kernel
    rms_matmul_rbe_kernel[grid](
        x_ptr=x, 
        weight_ptr=weight,
        output_ptr=output,
        rms_norm_ptr=rms_norm,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        head_dim=hidden_dim // 8,  # Assuming 8 heads
        stride_b=seq_len,
        stride_s=1,
        stride_h=hidden_dim,
        scale=scale,
        num_warps=4,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return output
