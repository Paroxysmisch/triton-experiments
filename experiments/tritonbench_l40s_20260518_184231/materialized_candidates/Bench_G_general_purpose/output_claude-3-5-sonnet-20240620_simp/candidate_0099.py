import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    # Pointers to matrices
    Out, # Output tensor pointer
    V,   # Values tensor pointer
    L,   # Logic tensor pointer
    M,   # Max values tensor pointer
    D,   # Denominator tensor pointer
    # Dimensions
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    # Strides
    stride_ob, stride_oh, stride_os, stride_od,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_lb, stride_lh, stride_ls,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute batch and head index
    batch_id = pid // num_heads
    head_id = pid % num_heads

    # Compute starting offsets for current batch and head
    off_o = batch_id * stride_ob + head_id * stride_oh
    off_v = batch_id * stride_vb + head_id * stride_vh
    off_l = batch_id * stride_lb + head_id * stride_lh

    # Load sequence length for current batch
    curr_seq_len = seq_len

    # Block level processing
    for block_start in range(0, curr_seq_len, BLOCK_SIZE):
        block_size = min(BLOCK_SIZE, curr_seq_len - block_start)
        
        # Initialize accumulators for current block
        acc = tl.zeros([head_dim], dtype=tl.float32)
        
        # Load values and logic for current block
        v_ptrs = off_v + block_start * stride_vs + tl.arange(0, block_size)[:, None] * stride_vd
        l_ptrs = off_l + block_start * stride_ls + tl.arange(0, block_size)
        
        v = tl.load(V + v_ptrs)
        l = tl.load(L + l_ptrs)
        m = tl.load(M + off_o + block_start)
        d = tl.load(D + off_o + block_start)
        
        # Compute scaling factor
        scale = tl.exp(m - d)
        
        # Accumulate weighted values
        acc += tl.sum(v * (l * scale)[:, None], axis=0)
    
    # Store final result
    o_ptrs = off_o + tl.arange(0, head_dim) * stride_od
    tl.store(Out + o_ptrs, acc)

def flash_decode_stage2(v, l, m, d):
    """
    Wrapper function for flash decode stage 2 kernel
    
    Args:
        v: Values tensor [batch_size, num_heads, seq_len, head_dim]
        l: Logic tensor [batch_size, num_heads, seq_len]
        m: Max values tensor [batch_size, num_heads, seq_len]
        d: Denominator tensor [batch_size, num_heads, seq_len]
    
    Returns:
        out: Output tensor [batch_size, num_heads, head_dim]
    """
    batch_size, num_heads, seq_len, head_dim = v.shape
    
    # Allocate output tensor
    out = torch.empty((batch_size, num_heads, head_dim), 
                     device=v.device, dtype=v.dtype)
    
    # Calculate strides
    stride_ob, stride_oh, stride_od = out.stride()
    stride_vb, stride_vh, stride_vs, stride_vd = v.stride()
    stride_lb, stride_lh, stride_ls = l.stride()
    
    # Configure kernel parameters
    BLOCK_SIZE = 128
    grid = (batch_size * num_heads,)
    
    # Launch kernel
    _fwd_kernel_flash_decode_stage2[grid](
        out, v, l, m, d,
        batch_size, num_heads, seq_len, head_dim,
        stride_ob, stride_oh, stride_os, stride_od,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_lb, stride_lh, stride_ls,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
