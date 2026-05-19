import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_token_softmax(
    Logics, Prob_Out,
    B_Start_Loc, B_Seqlen, max_input_len,
    stride_logic_h, stride_logic_bs,
    stride_prob_bs, stride_prob_h, stride_prob_d,
    stride_b_loc_b, stride_b_loc_s,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    off_logic = cur_head * stride_logic_h + (cur_batch_start_loc + offs_n) * stride_logic_bs
    off_prob = cur_batch * stride_prob_bs + cur_head * stride_prob_h + offs_d * stride_prob_d

    e_max = float("-inf")
    e_sum = 0.0

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        qk = tl.load(Logics + off_logic + start_n * stride_logic_bs, 
                     mask=start_n + offs_n < cur_batch_seq_len, other=float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 0), e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        e_sum = e_sum * old_scale + tl.sum(p, 0)
        e_max = n_e_max

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        qk = tl.load(Logics + off_logic + start_n * stride_logic_bs, 
                     mask=start_n + offs_n < cur_batch_seq_len, other=float("-inf"))
        p = tl.exp(qk - e_max) / e_sum
        tl.store(Prob_Out + off_prob + start_n * stride_prob_bs, p, mask=start_n + offs_n < cur_batch_seq_len)

    return

@torch.no_grad()
def token_softmax_fwd(logics, b_start_loc, b_seq_len, prob_out, max_input_len):
    BLOCK = 64
    batch, head = b_seq_len.shape[0], logics.shape[0]
    grid = (batch, head)

    num_warps = 1
    _fwd_kernel_token_softmax[grid](
        logics, prob_out,
        b_start_loc, b_seq_len, max_input_len,
        logics.stride(0), logics.stride(1),
        prob_out.stride(0), prob_out.stride(1), prob_out.stride(2),
        b_start_loc.stride(0), b_start_loc.stride(1),
        BLOCK_DMODEL=logics.shape[-1],
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=3
    )
    return
