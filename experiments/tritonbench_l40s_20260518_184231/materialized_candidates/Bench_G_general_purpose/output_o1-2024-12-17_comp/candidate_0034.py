import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att2(
    Prob_ptr, 
    V_ptr, 
    Out_ptr,
    Req_to_tokens_ptr,
    req_seq_offsets_ptr,
    kv_head_offsets_ptr,
    out_head_offsets_ptr,
    out_seq_offsets_ptr,
    stride_prob_head, 
    stride_prob_seq, 
    stride_prob_dim,
    stride_v_head, 
    stride_v_seq, 
    stride_v_dim,
    stride_out_head,
    stride_out_seq,
    stride_out_dim,
    total_heads, 
    total_seqs, 
    dim,
    BLOCK_N: tl.constexpr
):
    # program_id(0) identifies the current head
    cur_head = tl.program_id(0)
    # program_id(1) identifies the request index
    req_idx = tl.program_id(1)

    # offset for the current request in Req_to_tokens
    req_offset = tl.load(req_seq_offsets_ptr + req_idx)
    # the start token index for this request
    start_tok = tl.load(Req_to_tokens_ptr + req_offset)
    # the next request offset is used to calculate how many tokens belong to this request
    next_req_offset = tl.load(req_seq_offsets_ptr + req_idx + 1)
    # total tokens for current request
    req_num_tokens = next_req_offset - req_offset

    # offset in the key-value head array
    head_offset_kv = tl.load(kv_head_offsets_ptr + cur_head)
    # offset in the output head array
    head_offset_out = tl.load(out_head_offsets_ptr + cur_head)

    # each block in the N dimension processes BLOCK_N tokens
    # block base index
    n_block_idx = tl.arange(0, BLOCK_N)
    
    # each block in the M dimension (of the requests) is just the single request here
    # we accumulate into an array "acc" of size [dim]
    acc = tl.zeros((dim,), dtype=tl.float32)

    # loop over all tokens in the request, in BLOCK_N steps
    # each iteration processes a slice of tokens from [n_offset, n_offset + BLOCK_N)
    offs = 0
    while offs < req_num_tokens:
        # get token indices in [offs, offs + BLOCK_N)
        read_offs = offs + n_block_idx
        # mask out-of-bound tokens
        mask = read_offs < req_num_tokens

        # gather probability from Prob
        prob_offset = (head_offset_kv +   # shift by the offset of the current head in Prob
                       (req_idx * stride_prob_seq) +   # shift by request index
                       read_offs * stride_prob_dim)    # shift by tokens
        p_value = tl.where(
            mask,
            tl.load(Prob_ptr + prob_offset, mask=mask, other=0.0),
            0.0
        )

        # gather Value from V
        # each token read_offs[t] belongs to the global token index start_tok + read_offs[t]
        global_tok = start_tok + read_offs
        v_offset = (head_offset_kv * stride_v_head) + \
                   (global_tok * stride_v_seq)
        v_value = tl.load(V_ptr + v_offset[:, None] + tl.arange(0, dim) * stride_v_dim, mask=mask[:, None], other=0.0)

        # multiply prob and value
        # p_value shape is [BLOCK_N], v_value shape is [BLOCK_N, dim]
        # each row of v_value is scaled by p_value
        scaled_v = v_value * p_value[:, None]
        acc += tl.sum(scaled_v, axis=0)

        offs += BLOCK_N

    # write the results to out
    out_offset = (head_offset_out * stride_out_head) + \
                 (req_idx * stride_out_seq)
    tl.store(Out_ptr + out_offset + tl.arange(0, dim) * stride_out_dim, acc)


@torch.no_grad()
def token_att_fwd2(
    Prob, 
    V, 
    Out, 
    Req_to_tokens,
    req_seq_offsets,
    kv_head_offsets,
    out_head_offsets,
    out_seq_offsets,
    total_heads, 
    total_seqs, 
    dim,
    BLOCK=128
):
    """
    Prob:      [ total_heads x total_seqs x dim_prob ] probabilities
    V:         [ total_heads x total_global_tokens x dim ] values
    Out:       [ total_heads x total_seqs x dim ] output
    Req_to_tokens:    tensor mapping requests to all token indices
    req_seq_offsets:  offset of each request in Req_to_tokens
    kv_head_offsets:  offset for each head in Prob/V
    out_head_offsets: offset for each head in Out
    out_seq_offsets:  offset for each request used in indexing
    total_heads:      total heads
    total_seqs:       total number of requests
    dim:              dimension size
    BLOCK:            block size for parallel token processing
    """

    # grid: (heads, requests)
    grid = (total_heads, total_seqs)

    triton.run(
        _fwd_kernel_token_att2,
        grid=grid,
        num_warps=4,
        num_stages=2,
        # data pointers
        Prob_ptr=Prob.data_ptr(),
        V_ptr=V.data_ptr(),
        Out_ptr=Out.data_ptr(),
        Req_to_tokens_ptr=Req_to_tokens.data_ptr(),
        req_seq_offsets_ptr=req_seq_offsets.data_ptr(),
        kv_head_offsets_ptr=kv_head_offsets.data_ptr(),
        out_head_offsets_ptr=out_head_offsets.data_ptr(),
        out_seq_offsets_ptr=out_seq_offsets.data_ptr(),
        # strides for Prob
        stride_prob_head=Prob.stride(0),
        stride_prob_seq=Prob.stride(1),
        stride_prob_dim=Prob.stride(2),
        # strides for V
        stride_v_head=V.stride(0),
        stride_v_seq=V.stride(1),
        stride_v_dim=V.stride(2),
        # strides for Out
        stride_out_head=Out.stride(0),
        stride_out_seq=Out.stride(1),
        stride_out_dim=Out.stride(2),
        # sizes
        total_heads=total_heads,
        total_seqs=total_seqs,
        dim=dim,
        # compile-time constants
        BLOCK_N=BLOCK
    )
