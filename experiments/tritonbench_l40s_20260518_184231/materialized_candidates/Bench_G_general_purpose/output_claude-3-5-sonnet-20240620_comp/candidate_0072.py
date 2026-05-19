import triton
import triton.language as tl
import torch

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_inter(
    q_ptr, k_ptr, g_ptr, A_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_gz, stride_gh, stride_gm, stride_gk,
    stride_Az, stride_Ah, stride_Am, stride_An,
    Z, H, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    pid_in_group = pid % num_pid_in_group
    i_b = (pid_in_group // num_pid_n) * BLOCK_M
    j_b = (pid_in_group % num_pid_n) * BLOCK_N
    
    # Initialize pointers to the start of each block
    q_block_ptr = q_ptr + group_id * stride_qz
    k_block_ptr = k_ptr + group_id * stride_kz
    g_block_ptr = g_ptr + group_id * stride_gz
    A_block_ptr = A_ptr + group_id * stride_Az
    
    # Create block accumulators
    b_A = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Iterate over k dimension
    for k in range(0, K, BLOCK_K):
        # Load q block
        q_offs = (i_b + tl.arange(0, BLOCK_M)) * stride_qm + \
                 (k + tl.arange(0, BLOCK_K)) * stride_qk
        mask_q = (i_b + tl.arange(0, BLOCK_M)) < M
        b_q = tl.load(q_block_ptr + q_offs, mask=mask_q[:, None], other=0.0)
        
        # Load k block
        k_offs = (j_b + tl.arange(0, BLOCK_N)) * stride_kn + \
                 (k + tl.arange(0, BLOCK_K)) * stride_kk
        mask_k = (j_b + tl.arange(0, BLOCK_N)) < N
        b_k = tl.load(k_block_ptr + k_offs, mask=mask_k[None, :], other=0.0)
        
        # Load g block
        g_offs = (i_b + tl.arange(0, BLOCK_M)) * stride_gm + \
                 (k + tl.arange(0, BLOCK_K)) * stride_gk
        b_g = tl.load(g_block_ptr + g_offs, mask=mask_q[:, None], other=0.0)
        
        # Compute attention scores
        b_A += tl.dot(b_g * b_q, b_k)
    
    # Scale and apply softmax
    b_A = b_A / tl.sqrt(float(K))
    b_A = tl.exp(b_A)
    
    # Store results
    A_offs = (i_b + tl.arange(0, BLOCK_M)) * stride_Am + \
             (j_b + tl.arange(0, BLOCK_N)) * stride_An
    mask_A = (i_b + tl.arange(0, BLOCK_M))[:, None] < M & \
             (j_b + tl.arange(0, BLOCK_N))[None, :] < N
    tl.store(A_block_ptr + A_offs, b_A, mask=mask_A)

def chunk_fwd_intra_gated_gk_fn(q, k, g):
    """
    Wrapper function for the forward pass of gated attention computation
    """
    batch_size, n_heads, seq_len, d_head = q.shape
    
    # Output tensor
    A = torch.empty((batch_size, n_heads, seq_len, seq_len), 
                   device=q.device, dtype=q.dtype)
    
    # Configure block sizes
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    
    # Launch kernel
    grid = lambda meta: (
        batch_size * n_heads * triton.cdiv(seq_len, BLOCK_M) * triton.cdiv(seq_len, BLOCK_N),
    )
    
    chunk_gla_fwd_A_kernel_intra_sub_inter[grid](
        q, k, g, A,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        g.stride(0), g.stride(1), g.stride(2), g.stride(3),
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        batch_size, n_heads, seq_len, seq_len, d_head,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    return A
