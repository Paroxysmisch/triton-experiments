import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att1(
    Q,
    K,
    B_Loc,
    B_Start_Loc,
    B_Seqlen,
    Att_Out,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_loc_b,
    stride_loc_s,
    kv_group_num,
    BLOCK_n: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    block_start_loc = tl.program_id(2) * BLOCK_n

    offs_n = block_start_loc + tl.arange(0, BLOCK_n)

    off_q = (
        cur_batch_in_all_start_index * stride_qbs
        + cur_head * stride_qh
        + tl.arange(0, BLOCK_N)
    )
    off_k = (
        tl.arange(0, BLOCK_N) * stride_kbs
        + cur_kv_head * stride_kh
        + tl.arange(0, BLOCK_N)
    )

    q = tl.load(Q + off_q, mask=offs_n[None, :] < cur_batch_seq_len, other=0.0).to(tl.float32)

    k_ptrs = K + off_k

    # initialize counter and sum
    ctr = tl.zeros([BLOCK_N], dtype=tl.float32)
    sum = tl.zeros([BLOCK_N], dtype=tl.float32)
    loc_sum = tl.zeros([BLOCK_N], dtype=tl.float32)

    mask = tl.where(offs_n < cur_batch_seq_len, 1, 0)
    for start_n in range(0, block_mask * BLOCK_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=(start_n + offs_n[None, :]) < cur_batch_seq_len,
            other=0.0,
        ).to(tl.float32)
        loc = tl.load(B_Loc + (cur_batch_in_all_start_index + start_n) * stride_loc_b + offs_n * stride_loc_s,
            mask, other=0).to(tl.float32)

        qk = tl.sum(q[None, :] * k[:, None], 1)
        qk = tl.where(offs_n[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        p = tl.exp(qk)
        ctr += tl.exp(loc - loc_sum[:, None]) * tl.sum(p, 1)
        sum += tl.exp(loc - loc_sum[:, None]) * tl.sum(p * loc[:, None], 1)
        loc_sum += tl.exp(loc - loc_sum)

    sum_ Broadcast= [ctr[None, :]]
    ctr_ Broadcast= [ctr[None, :]]
    off_o = cur_head * stride_qh + tl.arange(0, BLOCK_N)
    att_out_ptrs = Att_Out + off_o[None, :, None]
    tl.store(
        att_out_ptrs,
        tl.exp(B_Loc + cur_batch * stride_loc_b + offs_n * stride_loc_s)[None, :, None] * (sum[:, None] / ctr[:, None]),
        mask=offs_n[None, :, None] < cur_batch_seq_len,
        boundary_check=(0, 1, -1),
    )


def token_att_fwd(q, k, b_loc, b_start_loc, b_seq_len, att_out, max_input_len):
    batch, head = b_seq_len.shape[0], q.shape[1]
    assert q.shape[2] == k.shape[2]
    L = q.shape[2]
    assert L in {16, 32, 64, 128, 256}

    kv_group_num = q.shape[1] // k.shape[1]

    BLOCK = 128

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))

    num_warps = 4

    if L <= 64:
        _fwd_kernel_token_att1[grid](
            q,
            k,
            b_loc,
            b_start_loc,
            b_seq_len,
            att_out,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            b_loc.stride(0),
            b_loc.stride(1),
            kv_group_num,
            BLOCK_n=BLOCK,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        _fwd_kernel_token_att1[grid](
            q,
            k,
            b_loc,
            b_start_loc,
            b_seq_len,
            att_out,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            b_loc.stride(0),
            b_loc.stride(1),
            kv_group_num,
            BLOCK_n=BLOCK,
            num_warps=num_warps,
            num_stages=1,
        )
