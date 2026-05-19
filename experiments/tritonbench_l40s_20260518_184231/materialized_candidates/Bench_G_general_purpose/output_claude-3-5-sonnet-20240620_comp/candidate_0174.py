import torch
import triton
import triton.language as tl

# Triton kernel for token attention forward pass
@triton.jit
def _fwd_kernel_token_att1(
    # Pointers to matrices
    Q, K, Att_Out, B_Loc, B_Start_Loc, B_Seqlen,
    # Matrix dimensions
    batch_size, num_heads, head_dim, max_input_len,
    # Strides for the different matrices
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_ob, stride_oh, stride_om,
    # Scale for attention scores
    sm_scale,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(max_input_len, BLOCK_M)
    num_pid_n = tl.cdiv(max_input_len, BLOCK_N)
    
    # Block ID
    bid = pid // (num_heads * num_pid_n)
    hid = (pid % (num_heads * num_pid_n)) // num_pid_n
    idx_n = (pid % num_pid_n) * BLOCK_N

    # Initialize pointers to Q, K
    q_ptr = Q + bid * stride_qb + hid * stride_qh
    k_ptr = K + bid * stride_kb + hid * stride_kh
    
    # Load sequence information
    b_start = tl.load(B_Start_Loc + bid)
    b_seq_len = tl.load(B_Seqlen + bid)
    
    # Initialize output pointer
    o_ptr = Att_Out + bid * stride_ob + hid * stride_oh
    
    # Compute attention scores for this block
    for idx_m in range(0, b_seq_len, BLOCK_M):
        # Load Q block
        q_block_ptr = q_ptr + idx_m * stride_qm
        q = tl.load(q_block_ptr + tl.arange(0, BLOCK_M)[:, None] * stride_qm)
        
        # Load K block
        k_block_ptr = k_ptr + idx_n * stride_kn
        k = tl.load(k_block_ptr + tl.arange(0, BLOCK_N)[None, :] * stride_kn)
        
        # Compute attention scores
        scores = tl.dot(q, k) * sm_scale
        
        # Load attention mask based on B_Loc
        mask = tl.load(B_Loc + b_start + tl.arange(0, BLOCK_M)[:, None] * max_input_len + 
                      tl.arange(0, BLOCK_N)[None, :])
        
        # Apply mask and store results
        scores = tl.where(mask, scores, float("-inf"))
        out_ptr = o_ptr + idx_m * stride_om + idx_n * stride_kn
        tl.store(out_ptr + tl.arange(0, BLOCK_M)[:, None] * stride_om + 
                tl.arange(0, BLOCK_N)[None, :], scores)

# Wrapper function
def token_att_fwd(q, k, b_loc, b_start_loc, b_seqlen, max_input_len, sm_scale=None):
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Validate input dimensions
    assert q.shape == k.shape, "Query and Key tensors must have the same shape"
    assert b_loc.shape[0] == batch_size, "Batch size mismatch"
    
    # Initialize output tensor
    output = torch.empty((batch_size, num_heads, seq_len, seq_len), 
                        device=q.device, dtype=q.dtype)
    
    # Calculate scaling factor if not provided
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)
    
    # Calculate strides
    stride_qb = q.stride(0)
    stride_qh = q.stride(1)
    stride_qm = q.stride(2)
    stride_kb = k.stride(0)
    stride_kh = k.stride(1)
    stride_kn = k.stride(2)
    stride_ob = output.stride(0)
    stride_oh = output.stride(1)
    stride_om = output.stride(2)
    
    # Define block sizes
    BLOCK_M = 32
    BLOCK_N = 32
    
    # Calculate grid size
    num_warps = 4
    grid = (batch_size * num_heads * triton.cdiv(seq_len, BLOCK_N),)
    
    # Launch kernel
    _fwd_kernel_token_att1[grid](
        q, k, output, b_loc, b_start_loc, b_seqlen,
        batch_size, num_heads, head_dim, max_input_len,
        stride_qb, stride_qh, stride_qm,
        stride_kb, stride_kh, stride_kn,
        stride_ob, stride_oh, stride_om,
        sm_scale,
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
        num_warps=num_warps
    )
    
    return output
