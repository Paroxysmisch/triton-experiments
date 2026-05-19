import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    B_Start_Loc, B_Seqlen,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(B_Seqlen, BLOCK_M)
    num_pid_n = tl.cdiv(B_Seqlen, BLOCK_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, B_Seqlen - first_pid_m)
    pid_m = (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block pointers
    start_m = first_pid_m * BLOCK_M + pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    
    # Initialize pointers to Q, K, V
    q_ptrs = Q + (start_m * stride_qm + tl.arange(0, BLOCK_M)[:, None] * stride_qk + 
                  tl.arange(0, BLOCK_DMODEL)[None, :])
    k_ptrs = K + (start_n * stride_kn + tl.arange(0, BLOCK_N)[:, None] * stride_kk +
                  tl.arange(0, BLOCK_DMODEL)[None, :])
    v_ptrs = V + (start_n * stride_vn + tl.arange(0, BLOCK_N)[:, None] * stride_vk +
                  tl.arange(0, BLOCK_DMODEL)[None, :])
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Load Q block
    q = tl.load(q_ptrs, mask=tl.arange(0, BLOCK_M)[:, None] < B_Seqlen, other=0.0)
    
    # Sliding window attention loop
    for n in range(0, tl.cdiv(B_Seqlen, BLOCK_N)):
        start_n = n * BLOCK_N
        # Check if block is within window
        in_window = tl.abs(start_m - start_n) <= WINDOW_SIZE
        
        if in_window:
            # Load K, V blocks
            k = tl.load(k_ptrs, mask=tl.arange(0, BLOCK_N)[:, None] < B_Seqlen, other=0.0)
            v = tl.load(v_ptrs, mask=tl.arange(0, BLOCK_N)[:, None] < B_Seqlen, other=0.0)
            
            # Compute attention scores
            qk = tl.dot(q, tl.trans(k))
            qk = qk * sm_scale
            
            # Apply softmax
            qk = tl.softmax(qk, axis=1)
            
            # Compute attention output
            acc += tl.dot(qk, v)
        
        # Advance K, V pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
    
    # Store output
    out_ptrs = Out + (start_m * stride_om + tl.arange(0, BLOCK_M)[:, None] * stride_on +
                      tl.arange(0, BLOCK_DMODEL)[None, :])
    tl.store(out_ptrs, acc, mask=tl.arange(0, BLOCK_M)[:, None] < B_Seqlen)

def context_attention_fwd(q, k, v, sm_scale, window_size):
    batch_size, n_heads, seq_len, d_head = q.shape
    
    # Kernel constants
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_DMODEL = d_head
    
    # Compute grid
    grid = (batch_size * n_heads * triton.cdiv(seq_len, BLOCK_M),)
    
    # Launch kernel
    _fwd_kernel[grid](
        q, k, v, 
        torch.empty_like(q),  # output tensor
        torch.arange(batch_size, device=q.device) * seq_len,  # batch start locations
        torch.full((batch_size,), seq_len, device=q.device),  # sequence lengths
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
        WINDOW_SIZE=window_size,
    )
