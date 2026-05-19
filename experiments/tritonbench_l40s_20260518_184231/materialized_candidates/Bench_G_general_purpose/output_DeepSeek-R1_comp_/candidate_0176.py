import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_softmax(
    Logits, Out, SequenceLengths,
    stride_logit_b, stride_logit_h, stride_logit_s,
    stride_out_b, stride_out_h, stride_out_s,
    max_seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_len = tl.load(SequenceLengths + batch_idx)
    
    logit_start_ptr = Logits + batch_idx * stride_logit_b + head_idx * stride_logit_h
    out_start_ptr = Out + batch_idx * stride_out_b + head_idx * stride_out_h
    
    max_logit = -float('inf')
    for offset in range(0, max_seq_len, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < seq_len
        logits = tl.load(logit_start_ptr + cols * stride_logit_s, mask=mask, other=-float('inf'))
        current_max = tl.max(logits, axis=0)
        max_logit = tl.maximum(max_logit, current_max)
    
    sum_exp = 0.0
    for offset in range(0, max_seq_len, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < seq_len
        logits = tl.load(logit_start_ptr + cols * stride_logit_s, mask=mask, other=-float('inf'))
        exp_logits = tl.exp(logits - max_logit)
        sum_exp += tl.sum(exp_logits, axis=0)
    
    for offset in range(0, max_seq_len, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < seq_len
        logits = tl.load(logit_start_ptr + cols * stride_logit_s, mask=mask, other=-float('inf'))
        exp_logits = tl.exp(logits - max_logit)
        softmax_out = exp_logits / sum_exp
        tl.store(out_start_ptr + cols * stride_out_s, softmax_out, mask=mask)

def token_softmax_fwd(logits, sequence_lengths):
    assert logits.dim() == 3, "Logits must be 3D [batch, num_heads, seq_len]"
    batch, num_heads, seq_len = logits.shape
    device = logits.device
    
    max_seq_len = sequence_lengths.max().item()
    
    output = torch.empty_like(logits)
    
    BLOCK_SIZE = triton.next_power_of_2(max_seq_len)
    BLOCK_SIZE = max(min(BLOCK_SIZE, 4096), 16)
    
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    
    grid = (batch, num_heads)
    _fwd_kernel_token_softmax[grid](
        logits, output, sequence_lengths,
        logits.stride(0), logits.stride(1), logits.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        max_seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return output
