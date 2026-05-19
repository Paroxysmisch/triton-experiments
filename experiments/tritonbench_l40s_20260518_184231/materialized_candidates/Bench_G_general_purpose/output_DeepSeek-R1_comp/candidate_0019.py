import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_softmax(
    Logits, Prob_Out,
    B_Start_Loc, B_Seqlen,
    num_heads: tl.constexpr,
    stride_logits_b, stride_logits_h,
    stride_probs_b, stride_probs_h,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // num_heads
    head_idx = pid % num_heads

    start_loc = tl.load(B_Start_Loc + batch_idx)
    seq_len = tl.load(B_Seqlen + batch_idx)

    logits_ptr = Logits + (start_loc * stride_logits_b) + (head_idx * stride_logits_h)
    prob_out_ptr = Prob_Out + (start_loc * stride_probs_b) + (head_idx * stride_probs_h)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len

    row = tl.load(logits_ptr + offsets * stride_logits_b, mask=mask, other=-float('inf'))

    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    probs = numerator / denominator

    tl.store(prob_out_ptr + offsets * stride_probs_b, probs, mask=mask)

@torch.no_grad()
def token_softmax_fwd(logits, b_start_loc, b_seqlen, prob_out):
    assert logits.is_contiguous(), "Logits tensor must be contiguous"
    assert prob_out.is_contiguous(), "Prob_Out tensor must be contiguous"
    assert b_start_loc.is_contiguous(), "B_Start_Loc tensor must be contiguous"
    assert b_seqlen.is_contiguous(), "B_Seqlen tensor must be contiguous"
    
    batch_size = b_start_loc.size(0)
    num_heads = logits.size(1)

    if batch_size == 0 or num_heads == 0:
        return

    max_input_len = b_seqlen.max().item()
    BLOCK_SIZE = triton.next_power_of_2(max_input_len)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    elif BLOCK_SIZE >= 1024:
        num_warps = 8

    grid = (batch_size * num_heads, )

    _fwd_kernel_token_softmax[grid](
        logits, prob_out,
        b_start_loc, b_seqlen,
        num_heads,
        logits.stride(0), logits.stride(1),
        prob_out.stride(0), prob_out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
