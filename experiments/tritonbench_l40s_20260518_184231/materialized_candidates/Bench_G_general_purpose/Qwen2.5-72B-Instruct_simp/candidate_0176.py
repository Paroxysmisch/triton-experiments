import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_softmax(
    Logits,  # Input logits tensor
    B_Start_Loc,  # Start indices of sequences
    B_Seqlen,  # Sequence lengths
    Prob_Out,  # Output tensor for probabilities
    stride_logit_b,  # Stride for logits in batch dimension
    stride_logit_h,  # Stride for logits in head dimension
    stride_logit_s,  # Stride for logits in sequence dimension
    stride_prob_b,  # Stride for output in batch dimension
    stride_prob_h,  # Stride for output in head dimension
    stride_prob_s,  # Stride for output in sequence dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch and head indices
    bid = tl.program_id(0)
    hid = tl.program_id(1)

    # Get the start index and sequence length for the current batch
    start_loc = tl.load(B_Start_Loc + bid)
    seqlen = tl.load(B_Seqlen + bid)

    # Initialize the maximum value and sum for the current sequence
    max_val = float('-inf')
    sum_val = 0.0

    # Iterate over the sequence elements
    for i in range(start_loc, start_loc + seqlen):
        logit = tl.load(Logits + bid * stride_logit_b + hid * stride_logit_h + i * stride_logit_s)
        max_val = tl.max(max_val, logit)

    # Compute the exponentials and sum
    for i in range(start_loc, start_loc + seqlen):
        logit = tl.load(Logits + bid * stride_logit_b + hid * stride_logit_h + i * stride_logit_s)
        exp_val = tl.exp(logit - max_val)
        sum_val += exp_val

    # Normalize the probabilities
    for i in range(start_loc, start_loc + seqlen):
        logit = tl.load(Logits + bid * stride_logit_b + hid * stride_logit_h + i * stride_logit_s)
        exp_val = tl.exp(logit - max_val)
        prob = exp_val / sum_val
        tl.store(Prob_Out + bid * stride_prob_b + hid * stride_prob_h + i * stride_prob_s, prob)

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['B', 'H', 'S'],
)
@triton.jit
def _fwd_kernel_token_softmax(
    Logits,  # Input logits tensor
    B_Start_Loc,  # Start indices of sequences
    B_Seqlen,  # Sequence lengths
    Prob_Out,  # Output tensor for probabilities
    stride_logit_b,  # Stride for logits in batch dimension
    stride_logit_h,  # Stride for logits in head dimension
    stride_logit_s,  # Stride for logits in sequence dimension
    stride_prob_b,  # Stride for output in batch dimension
    stride_prob_h,  # Stride for output in head dimension
    stride_prob_s,  # Stride for output in sequence dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Get the batch and head indices
    bid = tl.program_id(0)
    hid = tl.program_id(1)

    # Get the start index and sequence length for the current batch
    start_loc = tl.load(B_Start_Loc + bid)
    seqlen = tl.load(B_Seqlen + bid)

    # Initialize the maximum value and sum for the current sequence
    max_val = float('-inf')
    sum_val = 0.0

    # Iterate over the sequence elements
    for i in range(start_loc, start_loc + seqlen):
        logit = tl.load(Logits + bid * stride_logit_b + hid * stride_logit_h + i * stride_logit_s)
        max_val = tl.max(max_val, logit)

    # Compute the exponentials and sum
    for i in range(start_loc, start_loc + seqlen):
        logit = tl.load(Logits + bid * stride_logit_b + hid * stride_logit_h + i * stride_logit_s)
        exp_val = tl.exp(logit - max_val)
        sum_val += exp_val

    # Normalize the probabilities
    for i in range(start_loc, start_loc + seqlen):
        logit = tl.load(Logits + bid * stride_logit_b + hid * stride_logit_h + i * stride_logit_s)
        exp_val = tl.exp(logit - max_val)
        prob = exp_val / sum_val
        tl.store(Prob_Out + bid * stride_prob_b + hid * stride_prob_h + i * stride_prob_s, prob)

def token_softmax_fwd(Logits, B_Start_Loc, B_Seqlen, Prob_Out):
    B, H, S = Logits.shape
    assert B == B_Start_Loc.shape[0]
    assert B == B_Seqlen.shape[0]

    # Launch the kernel
    grid = (B, H)
    _fwd_kernel_token_softmax[grid](
        Logits, B_Start_Loc, B_Seqlen, Prob_Out,
        Logits.stride(0), Logits.stride(1), Logits.stride(2),
        Prob_Out.stride(0), Prob_Out.stride(1), Prob_Out.stride(2),
        BLOCK_SIZE=256,
    )

# Example usage
B, H, S = 32, 8, 128
Logits = torch.randn((B, H, S), device='cuda')
B_Start_Loc = torch.randint(0, S, (B,), device='cuda')
B_Seqlen = torch.randint(1, S, (B,), device='cuda')
Prob_Out = torch.empty((B, H, S), device='cuda')

token_softmax_fwd(Logits, B_Start_Loc, B_Seqlen, Prob_Out)
print(Prob_Out)
