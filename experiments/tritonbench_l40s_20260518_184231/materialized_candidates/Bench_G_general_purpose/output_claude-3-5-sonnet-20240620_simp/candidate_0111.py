import triton
import triton.language as tl
import torch

@triton.jit
def _attn_fwd_kernel(
    # Pointers to matrices
    q_ptr, k_ptr, v_ptr, out_ptr, mask_ptr,
    # Matrix dimensions
    batch_size, seq_len_q, seq_len_k, num_heads, head_dim,
    # Strides for the different matrices
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    # Optional window size
    window_size: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Block dimensions
    num_block_m = tl.cdiv(seq_len_q, BLOCK_M)
    num_block_n = tl.cdiv(seq_len_k, BLOCK_N)
    
    # Block indices
    block_m = pid // num_block_n
    block_n = pid % num_block_n
    
    # Starting indices for this block
    start_m = block_m * BLOCK_M
    start_n = block_n * BLOCK_N
    
    # Offsets for this program
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Load mask if provided
    if mask_ptr is not None:
        mask = tl.load(mask_ptr + offs_m[:, None] * seq_len_k + offs_n[None, :])
    
    # Iterate over blocks in head dimension
    for d in range(0, head_dim, BLOCK_DMODEL):
        # Compute Q @ K^T
        q = tl.load(q_ptr + offs_m[:, None] * stride_qm + d * stride_qh)
        k = tl.load(k_ptr + offs_n[None, :] * stride_kn + d * stride_kh)
        
        # Compute attention scores
        scores = tl.dot(q, k.transpose())
        
        # Apply mask and scaling
        if mask_ptr is not None:
            scores = scores * mask
        scores = scores * (1.0 / tl.sqrt(float(head_dim)))
        
        # Apply softmax
        scores = tl.softmax(scores, axis=1)
        
        # Load values and compute weighted sum
        v = tl.load(v_ptr + offs_n[:, None] * stride_vn + d * stride_vh)
        acc += tl.dot(scores, v)
    
    # Store output
    tl.store(out_ptr + offs_m[:, None] * stride_om, acc)

# Python wrapper
def attention_forward(q, k, v, mask=None, window_size=None):
    """
    Forward pass for attention mechanism with optional sliding window
    
    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len_q, head_dim)
        k: Key tensor of shape (batch_size, num_heads, seq_len_k, head_dim)
        v: Value tensor of shape (batch_size, num_heads, seq_len_k, head_dim)
        mask: Optional attention mask
        window_size: Optional sliding window size
    """
    batch_size, num_heads, seq_len_q, head_dim = q.shape
    _, _, seq_len_k, _ = k.shape
    
    # Output tensor
    output = torch.empty_like(q)
    
    # Block sizes (can be tuned)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = 32
    
    # Grid size
    grid = (triton.cdiv(seq_len_q, BLOCK_M) * triton.cdiv(seq_len_k, BLOCK_N),)
    
    # Launch kernel
    _attn_fwd_kernel[grid](
        q, k, v, output, mask,
        batch_size, seq_len_q, seq_len_k, num_heads, head_dim,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        window_size=window_size if window_size is not None else seq_len_k,
    )
    
    return output
