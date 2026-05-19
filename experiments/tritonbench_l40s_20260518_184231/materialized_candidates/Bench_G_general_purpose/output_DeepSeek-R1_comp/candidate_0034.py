import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att2(
    # Pointers to tensors
    Prob_ptr, V_ptr, Out_ptr, Req_to_tokens_ptr,
    # Tensor strides
    stride_pb, stride_ph, stride_pm, stride_pl,
    stride_vb, stride_vh, stride_vm, stride_vn,
    stride_ob, stride_oh, stride_om, stride_on,
    stride_req_to_tokens,
    kv_group_num,
    head_dim,
    max_seq_len,
    # Meta-parameters
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    req_idx = tl.program_id(0)
    batch_head_idx = tl.program_id(1)
    
    # Calculate current batch and head group
    num_heads_per_group = kv_group_num
    cur_batch = batch_head_idx // (num_heads_per_group)
    cur_head = (batch_head_idx % num_heads_per_group) * kv_group_num
    
    # Load request token boundaries
    req_start = tl.load(Req_to_tokens_ptr + req_idx * stride_req_to_tokens)
    req_end = tl.load(Req_to_tokens_ptr + req_idx * stride_req_to_tokens + 1)
    seq_len = req_end - req_start
    
    # Base pointers for current batch and head
    prob_base = Prob_ptr + cur_batch * stride_pb + cur_head * stride_ph
    v_base = V_ptr + cur_batch * stride_vb + cur_head * stride_vh
    out_base = Out_ptr + cur_batch * stride_ob + cur_head * stride_oh
    
    # Process each query position
    for m in tl.range(0, seq_len):
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        
        # Process key blocks
        for n_block in range(0, tl.cdiv(seq_len, BLOCK_N)):
            n_start = n_block * BLOCK_N
            n_mask = n_start + tl.arange(0, BLOCK_N) < seq_len
            
            # Load probability block
            p_offsets = m * stride_pm + (n_start + tl.arange(0, BLOCK_N)) * stride_pl
            p = tl.load(prob_base + p_offsets, mask=n_mask, other=0.0)
            
            # Load value block
            v_offsets = (n_start + tl.arange(0, BLOCK_N)[:, None]) * stride_vm + tl.arange(0, BLOCK_D)[None, :] * stride_vn
            v = tl.load(v_base + v_offsets, mask=n_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < head_dim), other=0.0)
            
            # Accumulate
            acc += tl.sum(p[:, None] * v, axis=0)
        
        # Store result
        out_offsets = (req_start + m) * stride_om + tl.arange(0, BLOCK_D) * stride_on
        tl.store(out_base + out_offsets, acc.to(Out_ptr.dtype.element_ty), mask=tl.arange(0, BLOCK_D) < head_dim)

@torch.no_grad()
def token_att_fwd2(prob, v, out, Req_to_tokens, kv_cache_scale=None):
    # Validate dimensions
    batch, num_heads, seq_len, head_dim = v.shape
    num_reqs = Req_to_tokens.size(0)
    
    # Calculate key-value groups
    q_len = prob.size(1)
    kv_group_num = num_heads // q_len if q_len != 0 else 1
    batch_heads = batch * num_heads // kv_group_num
    
    # Configure kernel grid and block
    BLOCK_N = 128
    BLOCK_D = triton.next_power_of_2(head_dim)
    if BLOCK_D > 2048: BLOCK_D = 2048
    
    grid = (num_reqs, batch_heads)
    
    # Launch kernel
    _fwd_kernel_token_att2[grid](
        prob, v, out, Req_to_tokens,
        prob.stride(0), prob.stride(1), prob.stride(2), prob.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Req_to_tokens.stride(0),
        kv_group_num,
        head_dim,
        seq_len,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
