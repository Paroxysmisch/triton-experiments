import triton
import triton.language as tl

# Define compile-time constants
BLOCK_M = 128  # Block size for sequence dimension
BLOCK_DMODEL = 64  # Block size for model dimension
BLOCK_N = 128  # Block size for key/value dimension

@triton.jit
def _fwd_kernel(Q, K, V, Out, B_Start_Loc, B_Seqlen, sm_scale, 
                stride_qm, stride_qh, stride_qd,
                stride_km, stride_kh, stride_kd,
                stride_vm, stride_vh, stride_vd,
                stride_om, stride_oh, stride_od,
                BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr):
    # Program ID for batch and head dimensions
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    
    # Block index for sequence
    block_idx = tl.program_id(2)
    
    # Compute offsets
    start_q = B_Start_Loc[batch_id] + block_idx * BLOCK_M
    seqlen_q = B_Seqlen[batch_id]
    
    # Load Q block
    q_offset = start_q * stride_qm + head_id * stride_qh
    Q_block = tl.load(Q + q_offset + tl.arange(0, BLOCK_M)[:, None] * stride_qd + tl.arange(0, BLOCK_DMODEL)[None, :])
    
    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Iterate over key blocks
    for k_start in range(0, seqlen_q, BLOCK_N):
        # Load K and V blocks
        k_offset = k_start * stride_km + head_id * stride_kh
        K_block = tl.load(K + k_offset + tl.arange(0, BLOCK_N)[:, None] * stride_kd + tl.arange(0, BLOCK_DMODEL)[None, :])
        
        v_offset = k_start * stride_vm + head_id * stride_vh
        V_block = tl.load(V + v_offset + tl.arange(0, BLOCK_N)[:, None] * stride_vd + tl.arange(0, BLOCK_DMODEL)[None, :])
        
        # Compute attention scores
        qk = tl.dot(Q_block, K_block, trans_b=True) * sm_scale
        
        # Apply softmax in a numerically stable way
        max_qk = tl.max(qk, axis=1)
        qk = qk - max_qk[:, None]
        weights = tl.exp(qk)
        weights_sum = tl.sum(weights, axis=1)
        
        # Compute weighted value sum
        weighted_v = tl.dot(weights, V_block)
        
        # Normalize by the sum of weights
        weighted_v = weighted_v / weights_sum[:, None]
        
        # Accumulate results
        acc += weighted_v
    
    # Store results
    out_offset = start_q * stride_om + head_id * stride_oh
    tl.store(Out + out_offset + tl.arange(0, BLOCK_M)[:, None] * stride_od + tl.arange(0, BLOCK_DMODEL)[None, :], acc)

def context_attention_fwd(Q, K, V, B_Start_Loc, B_Seqlen, sm_scale):
    # Determine grid dimensions
    batch_size, num_heads, seqlen_q, d_model = Q.shape
    grid = (batch_size, num_heads, (seqlen_q + BLOCK_M - 1) // BLOCK_M)
    
    # Allocate output tensor
    Out = torch.empty_like(Q)
    
    # Launch Triton kernel
    _fwd_kernel[grid](
        Q, K, V, Out, B_Start_Loc, B_Seqlen, sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        Out.stride(0), Out.stride(1), Out.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_N=BLOCK_N
    )
    
    return Out
