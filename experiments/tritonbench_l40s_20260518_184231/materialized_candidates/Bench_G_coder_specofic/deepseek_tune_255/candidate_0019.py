import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional

@triton.jit
def _fwd_kernel_token_softmax(
    Logics,
    B_Start_Loc,
    B_Seqlen,
    Prob_Out,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    cur_batch_start_index = tl.load(B_Start_Loc + batch_id)
    cur_batch_seq_len = tl.load(B_Seqlen + batch_id)
    cur_batch_logits_start_ptr = (
        Logics
        + cur_batch_start_index * Logics.shape[1]
        + head_id * Logics.shape[2]
    )
    cur_batch_prob_start_ptr = (
        Prob_Out
        + cur_batch_start_index * Prob_Out.shape[1]
        + head_id * Prob_Out.shape[2]
    )

    offs = tl.arange(0, BLOCK_SIZE)
    logits_ptrs = cur_batch_logits_start_ptr + offs
    prob_ptrs = cur_batch_prob_start_ptr + offs
    seq_len_mask = offs < cur_batch_seq_len

    logits = tl.load(logits_ptrs, mask=seq_len_mask, other=-float("inf"))
    logits = logits - tl.max(logits, axis=0)
    numerator = tl.exp(logits)
    denominator = tl.sum(numerator, axis=0)
    probs = numerator / denominator

    tl.store(prob_ptrs, probs, mask=seq_len_mask)


@torch.no_grad()
def token_softmax_fwd(
    logits: Tensor,
    b_start_loc: Tensor,
    b_seqlen: Tensor,
    prob_out: Optional[Tensor] = None,
    max_input_len: int = 0,
) -> Tensor:
    """
    Applies the softmax function to logits.
    """
    if not isinstance(logits, Tensor):
        logits = torch.tensor(logits)
    if not isinstance(b_start_loc, Tensor):
        b_start_loc = torch.tensor(b_start_loc)
    if not isinstance(b_seqlen, Tensor):
        b_seqlen = torch.tensor(b_seqlen)

    batch_size, head_num = b_start_loc.shape[0], logits.shape[1]

    if max_input_len <= 1024:
        BLOCK_SIZE = max_input_len
    else:
        BLOCK_SIZE = 1024
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    _fwd_kernel_token_softmax[(batch_size, head_num)](
        logits,
        b_start_loc,
        b_seqlen,
        prob_out,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_ctas=1,
    )
    return prob_out
