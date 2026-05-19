import triton
import triton.language as tl
import torch

@triton.jit
def fwd_decay_cumsum_kernel(
    g_ptr, g_o_ptr,
    stride_g_b, stride_g_h, stride_g_t,
    T, B, H,
    inv_ln2,
    BLOCK_T: tl.constexpr,
):
    # Program ID
    pid_b = tl.program_id(0)  # Batch
    pid_h = tl.program_id(1)  # Head
    pid_t = tl.program_id(2)  # Time dimension block

    # Initialize cumulative decay
    cum_decay = tl.zeros([BLOCK_T], dtype=tl.float32)
    
    # Compute base pointers
    g_block_ptr = g_ptr + pid_b * stride_g_b + pid_h * stride_g_h
    g_o_block_ptr = g_o_ptr + pid_b * stride_g_b + pid_h * stride_g_h
    
    # Time offsets
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = offs_t < T
    
    # Load g block
    g_offs = g_block_ptr + offs_t * stride_g_t
    g = tl.load(g_offs, mask=mask, other=0.0)
    
    # Compute cumulative sum with decay
    g = g * inv_ln2
    for i in range(BLOCK_T):
        if i > 0:
            cum_decay = cum_decay + g[i-1]
        g_o = tl.exp(-cum_decay) * g
        # Store result
        if mask[i]:
            tl.store(g_o_block_ptr + i * stride_g_t, g_o)

def fwd_decay_cumsum(g, inv_ln2):
    """
    Forward pass for decay cumulative sum operation
    Args:
        g: Input tensor of shape (B, H, T)
        inv_ln2: Inverse of ln(2) constant
    Returns:
        g_o: Output tensor of shape (B, H, T)
    """
    B, H, T = g.shape
    g_o = torch.empty_like(g)
    
    # Compute strides
    stride_g_b = g.stride(0)
    stride_g_h = g.stride(1)
    stride_g_t = g.stride(2)
    
    # Define block size
    BLOCK_T = 128
    
    # Launch kernel
    grid = (B, H, triton.cdiv(T, BLOCK_T))
    fwd_decay_cumsum_kernel[grid](
        g_ptr=g, 
        g_o_ptr=g_o,
        stride_g_b=stride_g_b,
        stride_g_h=stride_g_h,
        stride_g_t=stride_g_t,
        T=T, B=B, H=H,
        inv_ln2=inv_ln2,
        BLOCK_T=BLOCK_T,
    )
    
    return g_o
