import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_token_softmax(
    Logics, Out,
    B_Loc, B_Start_Loc, B_Seqlen, max_input_len,
    stride_logic_h, stride_logic_bs,
    stride_obs, stride_oh,
    stride_b_loc_b, stride_b_loc_s,
    other_kv_index,
    kv_group_num,
    BLOCK_SIZE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)

    cur_kv_head = cur_head // kv_group_num

    offs_n = tl.arange(0, BLOCK_SIZE)

    e_max = float("-inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for start_n in range(0, cur_batch_seq_len, BLOCK_SIZE):
        start_n = tl.multiple_of(start_n, BLOCK_SIZE)
        logic_index = cur_batch_start_loc + start_n + offs_n
        mask = (logic_index < cur_batch_seq_len + cur_batch_start_loc) & (start_n + offs_n < cur_batch_seq_len)
        qk = tl.load(Logics + cur_kv_head * stride_logic_h + logic_index * stride_logic_bs, mask=mask, other=float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 0), e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        e_sum = e_sum * old_scale + tl.sum(p, 0)
        acc = acc * old_scale + tl.sum(p, 0) * mask.to(tl.float32)
        e_max = n_e_max

    e_sum = tl.where(cur_batch_seq_len + cur_batch_start_loc <= max_input_len, e_sum, 0.0)
    acc = acc / e_sum

    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_n
    off_l = cur_batch * stride_b_loc_b + (max_input_len - cur_batch_seq_len) * stride_b_loc_s
    loc_off = tl.load(B_Loc + off_l + offs_n, mask=offs_n < cur_batch_seq_len, other=other_kv_index)

    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_n < cur_batch_seq_len)
    return

@torch.no_grad()
def token_softmax_fwd(logics, o, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index):
    BLOCK = 32
    batch, head = b_seq_len.shape[0], logics.shape[0]
    grid = (batch, head)
    kv_group_num = logics.shape[0] // b_loc.shape[1]

    num_warps = 4 if BLOCK >= 64 else 8
    _fwd_kernel_token_softmax[grid](
        logics, o, b_loc, b_start_loc, b_seq_len, max_input_len,
        logics.stride(0), logics.stride(1),
        o.stride(0), o.stride(1),
        b_loc.stride(0), b_loc.stride(1),
        other_kv_index,
        kv_group_num,
        BLOCK_SIZE=BLOCK,
        num_warps=num_warps,
        num_stages=2
    )
    return
