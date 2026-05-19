import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=16),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=32),
    ],
    key=["max_seqlen"],
)
@triton.jit
def _fwd_kernel_token_softmax(
    probs_ptr,
    logits_ptr,
    seqlen_ptr,
    stride_ptr,
    stride_logit_batch,
    stride_logit_head,
    stride_logit_seqlen,
    stride_prob_batch,
    stride_prob_head,
    stride_prob_seqlen,
    batch,
    head,
    max_seqlen,
    BLOCK_SIZE: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_start = cur_index * head + cur_head
    cur_batch_end = cur_batch_start + 1

    cur_batch_idx = cur_batch_start // head
    cur_head_idx = cur_batch_start % head

    cur_batch_seqlen = tl.load(seqlen_ptr + cur_batch_idx)
    cur_batch_stride = tl.load(stride_ptr + cur_batch_idx)

    cur_batch_logit = logits_ptr + cur_batch_stride * stride_logit_batch + cur_head_idx * stride_logit_head
    cur_batch_prob = probs_ptr + cur_batch_stride * stride_prob_batch + cur_head_idx * stride_prob_head

    logits_col = tl.arange(0, BLOCK_SIZE)
    prob_row = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    max_logit = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float("inf")
    sum_exp_logit = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for i in range(0, cur_batch_seqlen, BLOCK_SIZE):
        logit_row = tl.load(cur_batch_logit + i * stride_logit_seqlen + logits_col, mask=logits_col < cur_batch_seqlen)
        cur_max_logit = tl.max(logit_row, axis=0)
        max_logit = tl.where(max_logit < cur_max_logit, cur_max_logit, max_logit)

        exp_logit = tl.exp(logit_row - max_logit)
        sum_logit = tl.sum(exp_logit, axis=0)
        sum_exp_logit = tl.where(i + BLOCK_SIZE <= cur_batch_seqlen, sum_exp_logit + sum_logit, sum_exp_logit)
        prob_row = tl.where(logits_col < cur_batch_seqlen, exp_logit / sum_logit, prob_row)
        logits_col += BLOCK_SIZE

    tl.store(cur_batch_prob + tl.arange(0, BLOCK_SIZE), prob_row, mask=tl.arange(0, BLOCK_SIZE) < cur_batch_seqlen)
    return


@torch.no_grad()
def token_softmax_fwd(
    logits: torch.Tensor,
    probs: torch.Tensor,
    seqlen: torch.Tensor,
    stride: torch.Tensor,
) -> None:
    """
    Argument:
        logits: (batch, head, seqlen, hidden_size)
        probs: (batch, head, seqlen, hidden_size)
        seqlen: (batch)
        stride: (batch)
    Return:
        None
    """
    batch, head, seqlen, _ = logits.shape
    max_seqlen = int(seqlen.max().item())
    BLOCK_SIZE = triton.next_power_of_2(max_seqlen)

    def grid(meta):
        return (triton.cdiv(batch * head, meta["BLOCK_SIZE"]), 1)

    _fwd_kernel_token_softmax[grid](
        probs,
        logits,
        seqlen,
        stride,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        probs.stride(0),
        probs.stride(1),
        probs.stride(2),
        batch,
        head,
        max_seqlen,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return
