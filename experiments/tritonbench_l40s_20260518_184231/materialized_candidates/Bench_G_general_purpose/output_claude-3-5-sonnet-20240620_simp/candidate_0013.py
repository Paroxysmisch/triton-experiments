import torch
import triton
import triton.language as tl

# Constants for block sizes - these can be tuned for performance
BLOCK_M = 128
BLOCK_N = 128
BLOCK_DMODEL = 64

@triton.jit
def _fwd_kernel(
    # Pointers to matrices
    Q, K, V, Out,
    # Matrix dimensions
    B_Start_Loc, B_Seqlen, B_req_idx,
    # Attention scale
    sm_scale,
    # Matrix strides
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_on,
    # Metadata
    batch_size, num_heads, seqlen_q, seqlen_k, head_dim, kv_group_num,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(seqlen_q, BLOCK_M)
    num_pid_n = tl.cdiv(seqlen_k, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    pid_m = (pid % num_pid_in_group) // num_pid_n
    pid_n = (pid % num_pid_in_group) % num_pid_n

    # Batch/head index
    batch_id = group_id // (num_heads // kv_group_num)
    head_id = group_id % (num_heads // kv_group_num)

    # Block pointers
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    
    # Initialize offsets
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Load sequence length for current batch
    b_start = tl.load(B_Start_Loc + batch_id)
    b_seqlen = tl.load(B_Seqlen + batch_id)
    b_req_idx = tl.load(B_req_idx + batch_id)

    # Compute Q block pointer
    q_ptrs = Q + (b_req_idx * stride_qb + 
                  head_id * stride_qh + 
                  offs_m[:, None] * stride_qm + 
                  offs_d[None, :])
    
    # Compute K block pointer
    k_ptrs = K + (b_start * stride_kb + 
                  head_id * stride_kh + 
                  offs_n[:, None] * stride_kn + 
                  offs_d[None, :])

    # Load Q and K blocks
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    k = tl.load(k_ptrs, mask=offs_n[:, None] < b_seqlen, other=0.0)

    # Compute attention scores
    scores = tl.dot(q, tl.trans(k))
    scores = scores * sm_scale

    # Apply softmax
    scores = tl.softmax(scores, axis=1)

    # Load V block
    v_ptrs = V + (b_start * stride_vb + 
                  head_id * stride_vh + 
                  offs_n[:, None] * stride_vn + 
                  offs_d[None, :])
    v = tl.load(v_ptrs, mask=offs_n[:, None] < b_seqlen, other=0.0)

    # Compute output
    out = tl.dot(scores, v)

    # Write output
    out_ptrs = Out + (b_req_idx * stride_ob + 
                      head_id * stride_oh + 
                      offs_m[:, None] * stride_on + 
                      offs_d[None, :])
    tl.store(out_ptrs, out, mask=offs_m[:, None] < seqlen_q)

def context_attention_fwd(q, k, v, b_start_loc, b_seqlen, b_req_idx, kv_group_num=1):
    """
    Forward pass for context attention mechanism.
    
    Args:
        q: Query tensor of shape (batch_size, num_heads, seqlen_q, head_dim)
        k: Key tensor of shape (batch_size, num_heads, seqlen_k, head_dim)
        v: Value tensor of shape (batch_size, num_heads, seqlen_k, head_dim)
        b_start_loc: Starting locations for each batch
        b_seqlen: Sequence lengths for each batch
        b_req_idx: Request indices for each batch
        kv_group_num: Number of key/value head groups
    """
    batch_size, num_heads, seqlen_q, head_dim = q.shape
    seqlen_k = k.shape[2]

    # Compute attention scale
    sm_scale = 1.0 / torch.sqrt(torch.tensor(head_dim, dtype=q.dtype))

    # Output tensor
    out = torch.empty_like(q)

    # Configure grid
    grid = (batch_size * (num_heads // kv_group_num) * 
            triton.cdiv(seqlen_q, BLOCK_M) * 
            triton.cdiv(seqlen_k, BLOCK_N),)

    # Launch kernel
    _fwd_kernel[grid](
        q, k, v, out,
        b_start_loc, b_seqlen, b_req_idx,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        batch_size, num_heads, seqlen_q, seqlen_k, head_dim, kv_group_num,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
    )

    return out
