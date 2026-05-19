import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att2(
    Prob_ptr, V_ptr, Out_ptr,
    Req_to_tokens_ptr,
    batch_seq_len, seq_len, num_heads, head_dim,
    batch_stride_p, head_stride_p,
    batch_stride_v, head_stride_v, seq_stride_v,
    batch_stride_o, head_stride_o, seq_stride_o,
    BLOCK_N: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    num_seq_chunks = tl.cdiv(seq_len, BLOCK_N)
    
    # Calculate current batch and head
    cur_batch = pid // (num_heads * num_seq_chunks)
    cur_head = (pid % (num_heads * num_seq_chunks)) // num_seq_chunks
    chunk_id = pid % num_seq_chunks
    
    # Calculate starting position for this chunk
    start_n = chunk_id * BLOCK_N
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_N, head_dim], dtype=tl.float32)
    
    # Get sequence length for current batch
    cur_batch_seq_len = tl.load(batch_seq_len + cur_batch)
    
    # Create offsets for loading
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, head_dim)
    
    # Mask for valid sequence positions
    mask = offs_n < cur_batch_seq_len
    
    # Load request-to-token mapping
    req_idx = tl.load(Req_to_tokens_ptr + cur_batch * seq_len + offs_n, mask=mask, other=0)
    
    # Compute base pointers for current batch and head
    prob_base = Prob_ptr + cur_batch * batch_stride_p + cur_head * head_stride_p
    v_base = V_ptr + cur_batch * batch_stride_v + cur_head * head_stride_v
    
    # Load and multiply probabilities with values
    for k in range(0, cur_batch_seq_len, BLOCK_N):
        k_offs = tl.arange(0, BLOCK_N) + k
        k_mask = k_offs < cur_batch_seq_len
        
        # Load probabilities
        p_value = tl.load(
            prob_base + req_idx[:, None] * seq_len + k_offs[None, :],
            mask=mask[:, None] & k_mask[None, :],
            other=0.0
        )
        
        # Load values
        v_value = tl.load(
            v_base + k_offs[:, None] * seq_stride_v + offs_d[None, :],
            mask=k_mask[:, None],
            other=0.0
        )
        
        # Compute attention
        acc += tl.dot(p_value, v_value)
    
    # Store results
    out_ptr = Out_ptr + cur_batch * batch_stride_o + cur_head * head_stride_o
    offs_seq = req_idx * seq_stride_o
    
    # Write output
    tl.store(
        out_ptr + offs_seq[:, None] + offs_d[None, :],
        acc.to(Out_ptr.dtype.element_ty),
        mask=mask[:, None]
    )

@torch.no_grad()
def token_att_fwd2(
    prob: torch.Tensor,
    v: torch.Tensor,
    req_to_tokens: torch.Tensor,
    batch_seq_len: torch.Tensor,
    kv_group_num: int = 1
) -> torch.Tensor:
    # Get dimensions
    batch_size = prob.shape[0]
    num_heads = prob.shape[1]
    seq_len = prob.shape[-1]
    head_dim = v.shape[-1]
    
    # Compute output
    out = torch.empty(
        (batch_size, num_heads, seq_len, head_dim),
        device=prob.device,
        dtype=v.dtype
    )
    
    # Calculate strides
    batch_stride_p = prob.stride(0)
    head_stride_p = prob.stride(1)
    
    batch_stride_v = v.stride(0)
    head_stride_v = v.stride(1)
    seq_stride_v = v.stride(2)
    
    batch_stride_o = out.stride(0)
    head_stride_o = out.stride(1)
    seq_stride_o = out.stride(2)
    
    # Define block size and grid
    BLOCK = 32
    grid = (batch_size * num_heads * triton.cdiv(seq_len, BLOCK),)
    
    # Launch kernel
    _fwd_kernel_token_att2[grid](
        prob, v, out,
        req_to_tokens,
        batch_seq_len, seq_len, num_heads, head_dim,
        batch_stride_p, head_stride_p,
        batch_stride_v, head_stride_v, seq_stride_v,
        batch_stride_o, head_stride_o, seq_stride_o,
        BLOCK_N=BLOCK
    )
    
    return out
