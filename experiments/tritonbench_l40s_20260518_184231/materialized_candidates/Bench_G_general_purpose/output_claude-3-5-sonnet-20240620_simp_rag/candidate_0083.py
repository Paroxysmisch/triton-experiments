import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, Out,
    stride_qz, stride_qh, stride_qm,
    stride_kz, stride_kh, stride_kn,
    stride_vz, stride_vh, stride_vn,
    stride_oz, stride_oh, stride_om,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # Program ID
    pid_m = tl.program_id(0)
    pid_batch_head = tl.program_id(1)
    
    # Batch and head indexing
    batch_id = pid_batch_head // Q.shape[1]
    head_id = pid_batch_head % Q.shape[1]
    
    # Initialize offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Compute pointer offsets for Q, K, V
    q_ptrs = Q + (batch_id * stride_qz + head_id * stride_qh + 
                  offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (batch_id * stride_kz + head_id * stride_kh + 
                  offs_n[None, :] * stride_kn + offs_d[:, None])
    v_ptrs = V + (batch_id * stride_vz + head_id * stride_vh + 
                  offs_n[:, None] * stride_vn + offs_d[None, :])
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    softmax_num = tl.zeros([BLOCK_M], dtype=tl.float32)
    softmax_den = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Load Q block
    q = tl.load(q_ptrs)
    
    # Loop over K,V blocks
    for block_n in range(0, K.shape[2], BLOCK_N):
        # Load K,V blocks
        k = tl.load(k_ptrs + block_n * stride_kn)
        v = tl.load(v_ptrs + block_n * stride_vn)
        
        # Load relative position bias
        b = tl.load(B0 + offs_m[:, None] + (block_n + offs_n[None, :]))
        
        # Compute attention scores
        scores = tl.dot(q, k) * scale + b
        
        # Apply softmax
        scores = tl.math.exp2(scores)
        softmax_num += tl.dot(scores, v)
        softmax_den += tl.sum(scores, axis=1)
    
    # Normalize and store output
    acc = softmax_num / softmax_den[:, None]
    out_ptrs = Out + (batch_id * stride_oz + head_id * stride_oh + 
                      offs_m[:, None] * stride_om + offs_d[None, :])
    tl.store(out_ptrs, acc)

def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, b0, out):
    # Get input dimensions
    batch_size, num_heads, seq_len, d_model = q.shape
    
    # Define block sizes based on GPU capability
    BLOCK_M = 128
    BLOCK_N = 128 
    BLOCK_DMODEL = d_model
    
    # Compute scale factor
    scale = 1.0 / (d_model ** 0.5)
    
    # Compute grid dimensions
    grid = (seq_len // BLOCK_M, batch_size * num_heads)
    
    # Launch kernel
    _fwd_kernel_aligned[grid](
        q, k, v, b0, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=8,
        num_stages=2
    )
