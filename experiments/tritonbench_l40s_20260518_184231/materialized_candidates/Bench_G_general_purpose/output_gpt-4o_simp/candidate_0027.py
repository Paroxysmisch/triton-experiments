import triton
import triton.language as tl

# Define constants for block sizes
BLOCK_M = 128
BLOCK_N = 128

@triton.jit
def _attn_fwd_inner(Q, K, V, acc, l_i, m_i, q_scale, k_scale, stride_q, stride_k, stride_v, stride_o, BLOCK_M, BLOCK_N, start_idx):
    pid = tl.program_id(axis=0)
    
    # Compute block indices
    row_start = pid * BLOCK_M
    col_start = start_idx

    # Load Q, K, V for the block
    q = tl.load(Q + row_start * stride_q, mask=row_start < Q.shape[0])
    k = tl.load(K + col_start * stride_k, mask=col_start < K.shape[1])
    v = tl.load(V + col_start * stride_v, mask=col_start < V.shape[1])

    # Compute scaled dot-product
    q_scaled = q * q_scale
    k_scaled = k * k_scale
    score = tl.dot(q_scaled, k_scaled)

    # Apply mask and softmax normalization
    m_i = tl.maximum(m_i, score)
    exp_score = tl.exp(score - m_i)
    l_i = l_i + exp_score
    acc = acc + tl.dot(exp_score, v)

    # Store results back
    tl.store(l_i, l_i)
    tl.store(m_i, m_i)
    tl.store(acc, acc)

@triton.jit
def _attn_fwd(Q, K, V, O, q_scale, k_scale, stride_q, stride_k, stride_v, stride_o, BLOCK_M, BLOCK_N):
    pid = tl.program_id(axis=0)
    
    # Initialize accumulation and normalization variables
    acc = tl.zeros([BLOCK_M, V.shape[1]], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M, 1], dtype=tl.float32)
    m_i = tl.full([BLOCK_M, 1], -float('inf'), dtype=tl.float32)

    # Iterate over columns of K, V
    for start_idx in range(0, K.shape[1], BLOCK_N):
        _attn_fwd_inner(Q, K, V, acc, l_i, m_i, q_scale, k_scale, stride_q, stride_k, stride_v, stride_o, BLOCK_M, BLOCK_N, start_idx)

    # Normalize accumulated values
    acc = acc / l_i
    tl.store(O, acc)

def forward(q, k, v, q_scale, k_scale):
    # Determine the grid size
    grid = (q.shape[0] // BLOCK_M,)

    # Define strides
    stride_q = q.stride(0)
    stride_k = k.stride(0)
    stride_v = v.stride(0)
    stride_o = q.stride(0)

    # Allocate output tensor
    o = torch.empty((q.shape[0], v.shape[1]), dtype=torch.float32, device=q.device)

    # Launch Triton kernel
    _attn_fwd[grid](q, k, v, o, q_scale, k_scale, stride_q, stride_k, stride_v, stride_o, BLOCK_M, BLOCK_N)

    return o
