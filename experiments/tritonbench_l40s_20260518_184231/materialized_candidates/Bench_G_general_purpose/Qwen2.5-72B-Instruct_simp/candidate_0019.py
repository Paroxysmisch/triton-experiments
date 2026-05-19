import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_softmax(
    Logits,  # Input logits
    Prob_Out,  # Output probabilities
    stride_logit_b,  # Stride for batch dimension in logits
    stride_logit_h,  # Stride for head dimension in logits
    stride_logit_s,  # Stride for sequence dimension in logits
    stride_prob_b,  # Stride for batch dimension in probabilities
    stride_prob_h,  # Stride for head dimension in probabilities
    stride_prob_s,  # Stride for sequence dimension in probabilities
    n_tokens,  # Number of tokens in each sequence
    BLOCK_SIZE: tl.constexpr  # Block size for processing
):
    # Get the batch and head indices
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # Compute the offset for the current batch and head
    logits_offset = batch_idx * stride_logit_b + head_idx * stride_logit_h
    prob_offset = batch_idx * stride_prob_b + head_idx * stride_prob_h

    # Load the logits for the current batch and head
    logits = tl.load(Logits + logits_offset + tl.arange(0, BLOCK_SIZE) * stride_logit_s, mask=tl.arange(0, BLOCK_SIZE) < n_tokens, other=-float('inf'))

    # Compute the max value for numerical stability
    max_val = tl.max(logits, axis=0)

    # Subtract the max value for numerical stability
    logits = logits - max_val

    # Compute the exponentials
    exp_logits = tl.exp(logits)

    # Compute the sum of exponentials
    sum_exp = tl.sum(exp_logits, axis=0)

    # Compute the softmax probabilities
    probs = exp_logits / sum_exp

    # Store the probabilities
    tl.store(Prob_Out + prob_offset + tl.arange(0, BLOCK_SIZE) * stride_prob_s, probs, mask=tl.arange(0, BLOCK_SIZE) < n_tokens)

import torch

def token_softmax_fwd(logits: torch.Tensor, prob_out: torch.Tensor, block_size: int):
    # Get the dimensions
    batch_size, n_heads, n_tokens = logits.shape

    # Define the grid and block dimensions
    grid = (batch_size, n_heads)
    block = (block_size,)

    # Define the strides
    stride_logit_b = logits.stride(0)
    stride_logit_h = logits.stride(1)
    stride_logit_s = logits.stride(2)
    stride_prob_b = prob_out.stride(0)
    stride_prob_h = prob_out.stride(1)
    stride_prob_s = prob_out.stride(2)

    # Launch the kernel
    _fwd_kernel_token_softmax[grid, block](
        logits, prob_out,
        stride_logit_b, stride_logit_h, stride_logit_s,
        stride_prob_b, stride_prob_h, stride_prob_s,
        n_tokens, BLOCK_SIZE=block_size
    )

# Example usage
if __name__ == "__main__":
    # Create a random logits tensor
    batch_size = 2
    n_heads = 3
    n_tokens = 4
    logits = torch.randn((batch_size, n_heads, n_tokens), device='cuda')

    # Create an output tensor for probabilities
    prob_out = torch.empty_like(logits)

    # Define the block size
    block_size = 128

    # Perform the forward softmax
    token_softmax_fwd(logits, prob_out, block_size)

    # Print the results
    print("Logits:")
    print(logits)
    print("Probabilities:")
    print(prob_out)
