import torch
import triton
import triton.language as tl

@triton.jit
def attention_fwd_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, out_ptr, h_ptr,
    # Matrix dimensions
    batch_size, num_heads, seq_len, head_dim,
    # Strides for the different matrices
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    # Block sizes
    BLOCK_SIZE_T: tl.constexpr, BLOCK_SIZE_D: tl.constexpr,
    # Scale factor for attention scores
    scale,
):
    # Program ID
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(seq_len, BLOCK_SIZE_T)
    
    # Block indices
    batch_idx = pid // (num_heads * num_blocks)
    head_idx = (pid % (num_heads * num_blocks)) // num_blocks
    block_idx = pid % num_blocks

    # Offsets for the current block
    start_s = block_idx * BLOCK_SIZE_T
    offs_s = start_s + tl.arange(0, BLOCK_SIZE_T)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    
    # Initialize pointers for the current batch and head
    q_block_ptr = q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_block_ptr = k_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_block_ptr = v_ptr + batch_idx * stride_vb + head_idx * stride_vh
    
    # Load query block
    q = tl.load(q_block_ptr + offs_s[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                mask=(offs_s[:, None] < seq_len) & (offs_d[None, :] < head_dim))
    
    # Initialize accumulators
    acc = tl.zeros([BLOCK_SIZE_T, BLOCK_SIZE_D], dtype=tl.float32)
    normalizer = tl.zeros([BLOCK_SIZE_T], dtype=tl.float32)
    
    # Loop over key/value blocks
    for block_k in range(0, num_blocks):
        start_k = block_k * BLOCK_SIZE_T
        offs_k = start_k + tl.arange(0, BLOCK_SIZE_T)
        
        # Load key and value blocks
        k = tl.load(k_block_ptr + offs_k[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                   mask=(offs_k[:, None] < seq_len) & (offs_d[None, :] < head_dim))
        v = tl.load(v_block_ptr + offs_k[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                   mask=(offs_k[:, None] < seq_len) & (offs_d[None, :] < head_dim))
        
        # Compute attention scores
        scores = tl.dot(q, k.transpose())
        scores = scores * scale
        
        # Apply softmax
        scores = tl.softmax(scores)
        
        # Update accumulator
        acc += tl.dot(scores, v)
        normalizer += tl.sum(scores, axis=1)
    
    # Store results
    offs_o = start_s + tl.arange(0, BLOCK_SIZE_T)
    out_block_ptr = out_ptr + batch_idx * stride_ob + head_idx * stride_oh
    
    # Write output
    tl.store(out_block_ptr + offs_o[:, None] * stride_os + offs_d[None, :] * stride_od,
             acc, mask=(offs_o[:, None] < seq_len) & (offs_d[None, :] < head_dim))
    
    # Optionally store intermediate tensor h
    if h_ptr is not None:
        tl.store(h_ptr + batch_idx * seq_len + head_idx * seq_len + offs_o,
                normalizer, mask=offs_o < seq_len)

class AttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale=None):
        # Check input dimensions
        batch_size, num_heads, seq_len, head_dim = q.shape
        assert k.shape == v.shape == (batch_size, num_heads, seq_len, head_dim)
        
        # Set scale if not provided
        if scale is None:
            scale = 1.0 / (head_dim ** 0.5)
            
        # Allocate output
        output = torch.empty_like(q)
        h = torch.empty((batch_size, num_heads, seq_len), device=q.device, dtype=q.dtype)
        
        # Configure block sizes
        BLOCK_SIZE_T = 32
        BLOCK_SIZE_D = 32
        
        # Compute grid dimensions
        num_blocks = triton.cdiv(seq_len, BLOCK_SIZE_T)
        grid = (batch_size * num_heads * num_blocks,)
        
        # Launch kernel
        attention_fwd_kernel[grid](
            q, k, v, output, h,
            batch_size, num_heads, seq_len, head_dim,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            BLOCK_SIZE_T, BLOCK_SIZE_D,
            scale,
        )
        
        return output

# Example usage
def attention(q, k, v, scale=None):
    return AttentionFunction.apply(q, k, v, scale)
