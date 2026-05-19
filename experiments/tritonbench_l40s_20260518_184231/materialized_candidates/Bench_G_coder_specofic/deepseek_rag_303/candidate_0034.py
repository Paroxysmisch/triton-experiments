import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att2(
    Prob,
    V,
    Out,
    Req_to_tokens,
    B_Start_Loc,
    B_Seqlen,
    stride_prob_batch,
    stride_prob_head,
    stride_prob_dim,
    stride_v_batch,
    stride_v_head,
    stride_v_dim,
    kv_group_num: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    off_prob = (
        cur_batch * stride_prob_batch
        + cur_kv_head * stride_prob_head
        + offs_n * stride_prob_dim
    )
    off_v = (
        cur_batch * stride_v_batch
        + cur_kv_head * stride_v_head
        + offs_n * stride_v_dim
    )
    # load data
    p_value = tl.load(
        Prob + off_prob,
        mask=(cur_batch_in_all_start_index + offs_n) < cur_batch_seq_len,
        other=0.0,
    )
    v_value = tl.load(
        V + off_v,
        mask=(cur_batch_in_all_start_index + offs_n) < cur_batch_seq_len,
        other=0.0,
    )
    # initialize pointer to l
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc += tl.sum(p_value * v_value, 0)
    # initialize pointers to output
    req_index = cur_batch * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptrs = Out + Req_to_tokens + req_index
    tl.store(out_ptrs, acc)

@torch.no_grad()
def token_att_fwd2(prob, v, out, req_to_tokens, b_start_loc, b_seq_len):
    if CUDA_CAPABILITY[0] >= 8:
        BLOCK = 128
    else:
        BLOCK = 64

    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128, 256}

    sm_scale = 1.0 / (Lq**0.5)
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = (batch, head)
    num_warps = 4 if Lk <= 64 else 8

    global cached_kernel
    if cached_kernel:
        cached_kernel(
            grid,
            num_warps,
            q,
            k,
            v,
            sm_scale,
            b_start_loc,
            b_seq_len,
            o,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            o.stride(0),
            o.stride(1),
        )
        return

    _fwd_kernel_token_att2[grid](
        prob,
        v,
        out,
        req_to_tokens,
        b_start_loc,
        b_seq_len,
        stride_prob_batch=prob.stride(0),
        stride_prob_head=prob.stride(1),
        stride_prob_dim=prob.stride(2),
        stride_v_batch=v.stride(0),
        stride_v_head=v.stride(1),
        stride_v_dim=v.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    cached_kernel = wrap_kernel_launcher(_fwd_kernel_token_att2)
