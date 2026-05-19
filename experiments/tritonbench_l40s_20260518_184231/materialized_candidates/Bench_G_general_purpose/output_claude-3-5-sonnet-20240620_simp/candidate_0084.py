import triton
import triton.language as tl
import torch

# Constants for block sizes
BLOCK_M = 128
BLOCK_N = 128
BLOCK_DMODEL = 64

@triton.jit
def _fwd_kernel_aligned(
    # Pointers to matrices
    Q, K, V, B0, Out,
    # Matrix dimensions
    stride_qm, stride_qh, stride_qd,    # Strides for Q
    stride_kn, stride_kh, stride_kd,    # Strides for K
    stride_vn, stride_vh, stride_vd,    # Strides for V
    stride_b0m, stride_b0n,            # Strides for B0
    stride_om, stride_oh, stride_od,    # Strides for Out
    # Scaling factor for attention
    scale,
    # Block dimensions
    M, N, D,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    """
    Kernel for computing scaled dot-product attention with relative position biases
    """
    # Program ID
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    
    # Block dimensions
    bm = pid * BLOCK_M
    bn = 0
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Pointers for current block
    q_ptrs = Q + bm * stride_qm + (bid % D) * stride_qd + (bid // D) * stride_qh
    k_ptrs = K + bn * stride_kn + (bid % D) * stride_kd + (bid // D) * stride_kh
    v_ptrs = V + bn * stride_vn + (bid % D) * stride_vd + (bid // D) * stride_vh
    b0_ptrs = B0 + bm * stride_b0m + bn * stride_b0n
    
    # Load block dimensions
    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    
    # Iterate over blocks in N dimension
    for n in range(0, N, BLOCK_N):
        # Load Q, K blocks
        q = tl.load(q_ptrs + m_offs[:, None] * stride_qm)
        k = tl.load(k_ptrs + n_offs[None, :] * stride_kn)
        
        # Compute attention scores
        scores = tl.dot(q, k.transpose())
        scores = scores * scale
        
        # Add relative position bias
        bias = tl.load(b0_ptrs + m_offs[:, None] * stride_b0m + n_offs[None, :] * stride_b0n)
        scores = scores + bias
        
        # Apply softmax
        scores = tl.math.exp2(scores - tl.max(scores, axis=1)[:, None])
        scores = scores / tl.sum(scores, axis=1)[:, None]
        
        # Load V block and compute weighted sum
        v = tl.load(v_ptrs + n_offs[:, None] * stride_vn)
        acc += tl.dot(scores, v)
        
        # Update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        b0_ptrs += BLOCK_N * stride_b0n
    
    # Store output
    out_ptrs = Out + bm * stride_om + (bid % D) * stride_od + (bid // D) * stride_oh
    tl.store(out_ptrs + m_offs[:, None] * stride_om, acc)

def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, b0, scale):
    """
    Wrapper function for the attention kernel
    """
    # Extract dimensions
    batch_size, num_heads, seq_len_q, d_head = q.shape
    _, _, seq_len_k, _ = k.shape
    
    # Validate input shapes
    assert k.shape == v.shape, "Key and value tensors must have the same shape"
    assert q.shape[-1] == k.shape[-1], "Q and K must have the same embedding dimension"
    assert b0.shape == (seq_len_q, seq_len_k), "Bias shape must match sequence lengths"
    
    # Initialize output tensor
    output = torch.empty_like(q)
    
    # Calculate grid dimensions
    grid = (triton.cdiv(seq_len_q, BLOCK_M), batch_size * num_heads)
    
    # Launch kernel
    _fwd_kernel_aligned[grid](
        q, k, v, b0, output,
        q.stride(2), q.stride(1), q.stride(3),    # Q strides
        k.stride(2), k.stride(1), k.stride(3),    # K strides
        v.stride(2), v.stride(1), v.stride(3),    # V strides
        b0.stride(0), b0.stride(1),               # B0 strides
        output.stride(2), output.stride(1), output.stride(3),  # Output strides
        scale,
        seq_len_q, seq_len_k, d_head,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
    )
    
    return output
