import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att2(
    Prob, V, Req_to_tokens, Out,
    B_req_idx, B_Start_Loc, B_Seqlen,
    stride_probs_b, stride_probs_h, stride_probs_s,
    stride_v_b, stride_v_h, stride_v_d,
    stride_rtokens_s,  # Stride for Req_to_tokens (1D)
    stride_out_b, stride_out_h, stride_out_d,
    kv_group_num,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    
    # Load batch metadata
    req_idx = tl.load(B_req_idx + cur_batch)
    start_loc = tl.load(B_Start_Loc + cur_batch)
    seq_len = tl.load(B_Seqlen + cur_batch)
    
    # Calculate kv_head
    kv_head = cur_head // kv_group_num
    
    # Offset for output
    off_out = cur_batch * stride_out_b + cur_head * stride_out_h
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_DMODEL,), dtype=tl.float32)
    
    # Loop over sequence in blocks of BLOCK_N
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        
        # Load probabilities
        probs_off = cur_batch * stride_probs_b + cur_head * stride_probs_h + offs_n
        p = tl.load(Prob + probs_off, mask=mask_n, other=0.0)
        
        # Load token indices from Req_to_tokens
        token_off = start_loc + offs_n
        token_idx = tl.load(Req_to_tokens + token_off, mask=mask_n, other=0)
        
        # Load V values
        v_off = token_idx[:, None] * stride_v_b + kv_head * stride_v_h + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_v_d
        v = tl.load(V + v_off, mask=mask_n[:, None] & (tl.arange(0, BLOCK_DMODEL)[None, :] < BLOCK_DMODEL), other=0.0)
        
        # Accumulate
        acc += tl.sum(p[:, None] * v, axis=0)
    
    # Store output
    off_d = tl.arange(0, BLOCK_DMODEL)
    tl.store(Out + off_out + off_d, acc, mask=off_d < BLOCK_DMODEL)

def token_att_fwd2(Prob, V, Out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, kv_group_num):
    assert V.size(-1) % 16 == 0, "d_model must be multiple of 16"
    BLOCK_DMODEL = 16
    BLOCK_N = 32
    batch_size, num_heads = Prob.shape[0], Prob.shape[1]
    grid = (batch_size, num_heads)
    
    # Strides for Prob
    stride_probs_b = Prob.stride(0)
    stride_probs_h = Prob.stride(1)
    stride_probs_s = Prob.stride(2)
    
    # Strides for V
    stride_v_b = V.stride(0)
    stride_v_h = V.stride(1)
    stride_v_d = V.stride(2)
    
    # Strides for Req_to_tokens (assumed 1D)
    stride_rtokens_s = Req_to_tokens.stride(0)
    
    # Strides for Out
    stride_out_b = Out.stride(0)
    stride_out_h = Out.stride(1)
    stride_out_d = Out.stride(2)
    
    num_warps = 4
    num_stages = 3
    
    _fwd_kernel_token_att2[grid](
        Prob, V, Req_to_tokens, Out,
        B_req_idx, B_Start_Loc, B_Seqlen,
        stride_probs_b, stride_probs_h, stride_probs_s,
        stride_v_b, stride_v_h, stride_v_d,
        stride_rtokens_s,
        stride_out_b, stride_out_h, stride_out_d,
        kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
