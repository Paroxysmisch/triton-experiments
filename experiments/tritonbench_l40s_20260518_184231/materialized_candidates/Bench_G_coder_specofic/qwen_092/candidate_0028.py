import triton
import triton.language as tl
import torch

@triton.jit
def rotary_kernel(X, COS, SIN, batch_size, seq_len, num_heads, head_dim, IS_VARLEN, INTERLEAVED, CONJUGATE):
    batch = tl.program_id(0) // (seq_len * num_heads)
    seq = (tl.program_id(0) // num_heads) % seq_len
    head = tl.program_id(0) % num_heads
    
    m = tl.program_id(1) * BLOCK_M
    k = tl.program_id(2) * BLOCK_K
    
    batch_idx = batch * seq_len * num_heads * head_dim
    seq_idx = seq * num_heads * head_dim
    head_idx = head * head_dim
    
    for i in range(m, m + BLOCK_M):
        for j in range(k, k + BLOCK_K):
            idx = batch_idx + seq_idx + head_idx + i * head_dim + j
            cos_idx = seq * head_dim + i
            sin_idx = seq * head_dim + j
            
            cos_val = COS[cos_idx]
            sin_val = SIN[sin_idx]
            
            if CONJUGATE:
                sin_val = -sin_val
            
            x0 = X[idx]
            x1 = X[idx + head_dim]
            
            X[idx] = x0 * cos_val - x1 * sin_val
            X[idx + head_dim] = x0 * sin_val + x1 * cos_val

@triton.jit
def launch_rotary_kernel(X, COS, SIN, batch_size, seq_len, num_heads, head_dim, IS_VARLEN, INTERLEAVED, CONJUGATE):
    idx = tl.program_id(0)
    if idx < batch_size * seq_len * num_heads:
        rotary_kernel(X, COS, SIN, batch_size, seq_len, num_heads, head_dim, IS_VARLEN, INTERLEAVED, CONJUGATE)

def apply_rotary(X, COS, SIN, batch_size, seq_len, num_heads, head_dim, IS_VARLEN=0, INTERLEAVED=0, CONJUGATE=0):
    assert X.is_contiguous()
    assert COS.is_contiguous()
    assert SIN.is_contiguous()
    
    if IS_VARLEN:
        grid = (batch_size * seq_len * num_heads, 1, 1)
    else:
        grid = (batch_size * seq_len * num_heads, seq_len // BLOCK_M, head_dim // BLOCK_K)
    
    launch_rotary_kernel[grid](X, COS, SIN, batch_size, seq_len, num_heads, head_dim, IS_VARLEN, INTERLEAVED, CONJUGATE)
    return X
