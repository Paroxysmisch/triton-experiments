import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def _fwd_kernel_token_softmax(
    Logics, B_Start_Loc, B_Seqlen,
    Prob_Out,
    stride_b, stride_h, stride_s,
    stride_bo, stride_ho, stride_so,
    BLOCK_SIZE: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_s = tl.arange(0, BLOCK_SIZE)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    cur_batch_logic_off = cur_batch * stride_b + cur_head * stride_h + offs_s * stride_s
    cur_batch_prob_off = cur_batch * stride_bo + cur_head * stride_ho + offs_s * stride_so

    logic_block_ptr = Logics + cur_batch_logic_off
    prob_block_ptr = Prob_Out + cur_batch_prob_off

    mask = offs_s < cur_batch_seq_len

    cur_batch_all_start_index = cur_batch_in_all_start_index + offs_s
    cur_batch_logic = tl.load(logic_block_ptr, mask=mask, other=-float('inf'))

    # for stable
    cur_batch_logic_max = tl.max(cur_batch_logic, axis=0)
    cur_batch_logic = cur_batch_logic - cur_batch_logic_max

    # softmax
    numerator = tl.exp(cur_batch_logic)
    denominator = tl.sum(numerator, axis=0)
    cur_batch_prob = numerator / denominator

    tl.store(prob_block_ptr, cur_batch_prob, mask=mask)

@torch.no_grad()
def token_softmax_fwd(Logics: Tensor, B_Start_Loc: Tensor, B_Seqlen: Tensor, max_input_len: int) -> Tensor:
    _, seq_len = B_Start_Loc.shape[0], B_Seqlen.shape[-1]

    assert Logics.shape[-1] == B_Seqlen.shape[-1]
    assert B_Start_Loc.shape[0] == B_Seqlen.shape[0]

    Prob_Out = torch.empty_like(Logics)

    BLOCK_SIZE = triton.next_power_of_2(max_input_len)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    grid = (B_Start_Loc.shape[0], 1)
    _fwd_kernel_token_softmax[grid](
        Logics, B_Start_Loc, B_Seqlen,
        Prob_Out,
        Logics.stride(0), Logics.stride(1), Logics.stride(2),
        Prob_Out.stride(0), Prob_Out.stride(1), Prob_Out.stride(2),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )
    return Prob_Out
