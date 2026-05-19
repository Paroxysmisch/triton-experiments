import triton
import triton.language as tl

@triton.jit
def block_sparse_attention_kernel(
    Q, K, V, out, 
    layout_csr_row_indices, layout_csr_col_indices,
    softmax_scale,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    num_heads, num_kv_heads,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr
):
    # Determine the current batch, head, and query block
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)
    
    cur_kv_head = cur_head // (num_heads // num_kv_heads)
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Offsets for Q, K, V
    off_q = (cur_batch * stride_qbs + cur_head * stride_qh + offs_m[:, None] * stride_qd)
    off_k = (cur_batch * stride_kbs + cur_kv_head * stride_kh + offs_n[None, :] * stride_kd)
    off_v = (cur_batch * stride_vbs + cur_kv_head * stride_vh + offs_n[:, None] * stride_vd)
    
    q = tl.load(Q + off_q)
    
    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    row_start = tl.load(layout_csr_row_indices + start_m)
    row_end = tl.load(layout_csr_row_indices + start_m + 1)
    
    for block_idx in range(row_start, row_end):
        col_idx = tl.load(layout_csr_col_indices + block_idx)
        
        k = tl.load(K + off_k + col_idx * BLOCK_N * stride_kd)
        v = tl.load(V + off_v + col_idx * BLOCK_N * stride_vd)
        
        qk = tl.dot(q, k)
        qk *= softmax_scale
        
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        acc += tl.dot(p, v)
        
        m_i = m_i_new
        l_i = l_i_new
    
    # Write results to output tensor
    off_o = (cur_batch * stride_obs + cur_head * stride_oh + offs_m[:, None] * stride_od)
    tl.store(out + off_o, acc)

def block_sparse_attention_fwd(Q, K, V, layout_csr_row_indices, layout_csr_col_indices, out, softmax_scale, num_heads, num_kv_heads, BLOCK_M, BLOCK_N, BLOCK_D):
    grid = (Q.shape[0], num_heads, triton.cdiv(Q.shape[2], BLOCK_M))
    
    block_sparse_attention_kernel[grid](
        Q, K, V, out,
        layout_csr_row_indices, layout_csr_col_indices,
        softmax_scale,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        num_heads, num_kv_heads,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        NUM_D_BLOCKS=1
    )
