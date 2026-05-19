import triton
import triton.language as tl
import torch

@triton.jit
def rms_matmul_rbe_kernel(
    # Pointers to matrices
    x_ptr, w_ptr, rms_w_ptr, out_ptr,
    # Matrix dimensions
    B, H, N, D,
    # Strides
    stride_xb, stride_xh, stride_xn,
    stride_wb, stride_wh, stride_wd,
    stride_ob, stride_oh, stride_on,
    # RBE (Rotary) parameters
    theta_base: float,
    use_rbe: bool,
    # Optional parameters
    USE_FP16: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Computes: out = (rms(x) * rms_w) @ w with optional rotary embeddings
    """
    pid = tl.program_id(0)
    
    # Compute batch/head/sequence indices
    batch_id = pid // (H * N)
    head_id = (pid % (H * N)) // N  
    seq_id = pid % N

    # Initialize pointers
    x_block_ptr = x_ptr + batch_id * stride_xb + head_id * stride_xh + seq_id * stride_xn
    w_block_ptr = w_ptr + batch_id * stride_wb + head_id * stride_wh
    
    # Load x block
    x = tl.load(x_block_ptr + tl.arange(0, BLOCK_SIZE_D))
    
    # Compute RMS norm
    rms = tl.sqrt(tl.mean(x * x) + 1e-6)
    rms_weight = tl.load(rms_w_ptr + head_id)
    x_normalized = x * rms * rms_weight
    
    # Apply rotary embeddings if enabled
    if use_rbe:
        position = seq_id
        freqs = tl.exp(-theta_base * tl.arange(0, BLOCK_SIZE_D//2))
        cos = tl.cos(position * freqs)
        sin = tl.sin(position * freqs)
        
        x_even = x_normalized[::2]
        x_odd = x_normalized[1::2]
        x_normalized = tl.concatenate([
            x_even * cos - x_odd * sin,
            x_odd * cos + x_even * sin
        ])
    
    # Matrix multiplication
    acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    for d in range(0, D, BLOCK_SIZE_D):
        w_block = tl.load(w_block_ptr + d + tl.arange(0, BLOCK_SIZE_D))
        acc += tl.dot(x_normalized, w_block)
    
    # Store output
    out_block_ptr = out_ptr + batch_id * stride_ob + head_id * stride_oh + seq_id * stride_on
    if USE_FP16:
        acc = acc.to(tl.float16)
    tl.store(out_block_ptr, acc)

# Wrapper function
def rms_matmul_rbe_qkv_wrapper(
    x: torch.Tensor,
    w: torch.Tensor,
    rms_w: torch.Tensor,
    use_rbe: bool = True,
    theta_base: float = 10000.0
):
    """
    Wrapper for RMS MatMul with optional rotary embeddings for Q,K,V attention
    Args:
        x: Input tensor of shape [B, H, N, D]
        w: Weight tensor of shape [B, H, D, D]
        rms_w: RMS weights of shape [H]
        use_rbe: Whether to use rotary embeddings
        theta_base: Base for rotary embedding frequencies
    Returns:
        Output tensor of shape [B, H, N, D]
    """
    batch_size, num_heads, seq_len, dim = x.shape
    
    # Allocate output
    output = torch.empty_like(x)
    
    # Configure grid and block sizes
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_D = 64
    
    grid = (batch_size * num_heads * seq_len,)
    
    # Launch kernel
    rms_matmul_rbe_kernel[grid](
        x, w, rms_w, output,
        batch_size, num_heads, seq_len, dim,
        x.stride(0), x.stride(1), x.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        theta_base,
        use_rbe,
        USE_FP16=x.dtype == torch.float16,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
    
    return output
