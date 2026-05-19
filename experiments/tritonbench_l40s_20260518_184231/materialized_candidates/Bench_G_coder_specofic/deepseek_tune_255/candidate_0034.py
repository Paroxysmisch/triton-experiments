import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att2(
    Prob, V, Out,  # inputs
    Req_to_tokens,
    #
    cur_batch,
    cur_head,
    seq_len,
    #
    BLOCK_N: tl.constexpr,
    #
    stride_req_batch,
    stride_req_head,
    stride_req_cand,
    stride_req_dim,
    #
    stride_batch_head,
    stride_batch_head_cand,
    stride_batch_head_dim,
    #
    stride_out_batch,
    stride_out_head,
    stride_out_cand,
    stride_out_dim,
    #
    kv_group_num: tl.constexpr,
    #
    BLOCK: tl.constexpr,
    #
    IS_CAUSAL: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)
    head_idx = tl.program_id(axis=1)
    batch_idx = cur_batch
    seq_start_idx = cur_head * BLOCK_N
    #
    req_batch_idx = req_idx // kv_group_num
    req_head_idx = req_idx % kv_group_num
    #
    off_req_batch = req_batch_idx * stride_req_batch
    off_req_head = req_head_idx * stride_req_head
    off_req_cand = off_req_head + stride_req_head * seq_len
    #
    off_batch_head = (batch_idx * stride_batch_head
                      + head_idx * stride_batch_head)
    off_batch_head_cand = off_batch_head + stride_batch_head_cand
    #
    off_out_batch = batch_idx * stride_out_batch
    off_out_head = head_idx * stride_out_head
    #
    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    #
    # initialize pointer to m and l
    m_ptr = tl.make_block_ptr(
        base=Prob + off_req_batch,
        shape=(seq_len, seq_len),
        strides=(stride_req_cand, stride_req_dim),
        offsets=(off_req_cand, off_req_head + seq_start_idx),
        block_shape=(BLOCK_N, 1),
        order=(1, 0),
    )
    l_ptr = tl.make_block_ptr(
        base=V + off_batch_head_cand,
        shape=(seq_len,),
        strides=(stride_batch_head_cand,),
        offsets=(0,),
        block_shape=(BLOCK_N,),
        order=(0,),
    )
    v_ptr = tl.make_block_ptr(
        base=V + off_batch_head,
        shape=(seq_len,),
        strides=(stride_batch_head_cand, stride_batch_head),
        offsets=(0, off_batch_head_cand),
        block_shape=(1, BLOCK_N),
        order=(1, 0),
    )
    # initialize pointer to output
    out_ptr = tl.make_block_ptr(
        base=Out + off_out_batch,
        shape=(seq_len,),
        strides=(stride_out_cand, stride_out_head, stride_out_dim),
        offsets=(0, off_out_head, 0),
        block_shape=(1,),
        order=(0,),
    )
    #
    acc = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
    #
    for start_n in range(0, seq_len, BLOCK_N):
        # bound the number of tokens
        end_n = min(start_n + BLOCK_N, seq_len)
        if end_n - start_n < BLOCK_N:
            mask_n = offs_n + start_n < end_n
        else:
            mask_n = offs_n + start_n < seq_len
        #
        p_value = tl.load(m_ptr, boundary_check=(0, 1)).to(tl.float32)
        if IS_CAUSAL:
            causal_mask = tl.where(offs_n + start_n >= (seq_start_idx + seq_len), 0, 1)
            p_value = p_value * causal_mask
        p_value = tl.where(mask_n, p_value, 0)
        #
        v_value = tl.load(v_ptr, boundary_check=(1,)).to(tl.float32)
        v_value = tl.where(mask_n[:, None], v_value, 0)
        #
        acc += tl.dot(p_value, v_value, allow_tf32=False)
        #
        l_value = tl.load(l_ptr, boundary_check=(0,)).to(tl.float32)
        l_value = tl.where(mask_n[None, :], l_value, 0)
        #
        acc = acc * l_value[None, :]
        #
        m_ptr = tl.advance(m_ptr, (BLOCK_N, 0))
        l_ptr = tl.advance(l_ptr, (0, BLOCK_N))
        v_ptr = tl.advance(v_ptr, (0, BLOCK_N))
    #
    acc = tl.sum(acc, axis=1)[:, None]
    #
    tl.store(out_ptr, acc.to(Out.dtype.element_ty), boundary_check=(0,))


@torch.no_grad()
def token_att_fwd2(prob, V, Out, Req_to_tokens, BLOCK: int = 32, IS_CAUSAL: bool = False):
    seq_len = V.shape[0]
    #
    kv_group_num = V.shape[1] // BLOCK
    batch_size = Req_to_tokens.shape[0]
    head_num = V.shape[2]
    #
    BLOCK_N = min(BLOCK, triton.next_power_of_2(seq_len))
    grid = (batch_size * head_num, 1)
    num_warps = 1
    #
    assert V.shape[1] == kv_group_num * BLOCK
    assert V.shape[2] % BLOCK == 0
    #
    V = V.view((seq_len // BLOCK) * BLOCK, -1, BLOCK)
    #
    _fwd_kernel_token_att2[grid](
        prob, V, Out,
        Req_to_tokens,
        #
        cur_batch=0,
        cur_head=0,
        seq_len=seq_len,
        #
        BLOCK_N=BLOCK_N,
        #
        stride_req_batch=Req_to_tokens.stride(0),
        stride_req_head=Req_to_tokens.stride(1),
        stride_req_cand=Req_to_tokens.stride(2),
        stride_req_dim=Req_to_tokens.stride(3),
        #
        stride_batch_head=V.stride(0),
        stride_batch_head_cand=V.stride(1),
        stride_batch_head_dim=V.stride(2),
        #
        stride_out_batch=Out.stride(0),
        stride_out_head=Out.stride(
