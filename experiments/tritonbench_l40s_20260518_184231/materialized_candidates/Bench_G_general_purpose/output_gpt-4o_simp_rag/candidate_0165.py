import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_token_softmax(
    Logics, Prob_Out,
    B_Start_Loc, B_Seqlen,
    stride_logic_h, stride_logic_bs,
    stride_out_h, stride_out_bs,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)

    offs_n = tl.arange(0, BLOCK_N)

    e_max = float("-inf")
    e_sum = 0.0

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        qk = tl.load(Logics + cur_head * stride_logic_h + (cur_batch_start_loc + start_n + offs_n) * stride_logic_bs,
                     mask=start_n + offs_n < cur_batch_seq_len, other=float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 0), e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        e_sum = e_sum * old_scale + tl.sum(p, 0)
        e_max = n_e_max

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        qk = tl.load(Logics + cur_head * stride_logic_h + (cur_batch_start_loc + start_n + offs_n) * stride_logic_bs,
                     mask=start_n + offs_n < cur_batch_seq_len, other=float("-inf"))
        p = tl.exp(qk - e_max) / e_sum
        tl.store(Prob_Out + cur_head * stride_out_h + (cur_batch_start_loc + start_n + offs_n) * stride_out_bs,
                 p, mask=start_n + offs_n < cur_batch_seq_len)

@torch.no_grad()
def token_softmax_fwd(logics, prob_out, b_start_loc, b_seq_len):
    BLOCK = 64
    batch, head = b_seq_len.shape[0], logics.shape[0]
    grid = (batch, head)

    _fwd_kernel_token_softmax[grid](
        logics, prob_out,
        b_start_loc, b_seq_len,
        logics.stride(0), logics.stride(1),
        prob_out.stride(0), prob_out.stride(1),
        BLOCK_N=BLOCK,
        num_warps=1,
        num_stages=3
    )
    return
