import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_softmax(
    logits_ptr,  # Pointer to the input logits (shape: [batch_size, num_heads, seq_len, seq_len])
    output_ptr,  # Pointer to the output softmax values (shape: [batch_size, num_heads, seq_len, seq_len])
    seq_lengths_ptr,  # Pointer to the sequence lengths (shape: [batch_size])
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the current batch and head indices
    pid = tl.program_id(axis=0)
    batch_idx = pid // num_heads
    head_idx = pid % num_heads

    # Load the sequence length for the current batch
    seq_length = tl.load(seq_lengths_ptr + batch_idx)

    # Compute the offset for the current batch and head
    offset = (batch_idx * num_heads + head_idx) * seq_len * seq_len

    # Iterate over the sequence length in blocks
    for block_start in range(0, seq_length, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, seq_length)

        # Load the logits for the current block
        logits = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
        for i in range(block_start, block_end):
            for j in range(block_start, block_end):
                if i < seq_length and j < seq_length:
                    logits[i - block_start, j - block_start] = tl.load(logits_ptr + offset + i * seq_len + j)

        # Compute the maximum logit value for numerical stability
        max_logit = tl.max(logits, axis=1)

        # Subtract the maximum logit value from each element
        logits = logits - max_logit[:, None]

        # Compute the exponentiated logits
        exp_logits = tl.exp(logits)

        # Compute the sum of exponentiated logits
        sum_exp_logits = tl.sum(exp_logits, axis=1)

        # Compute the softmax values
        softmax_values = exp_logits / sum_exp_logits[:, None]

        # Store the softmax values back to the output pointer
        for i in range(block_start, block_end):
            for j in range(block_start, block_end):
                if i < seq_length and j < seq_length:
                    tl.store(output_ptr + offset + i * seq_len + j, softmax_values[i - block_start, j - block_start])

    # Mask invalid positions with negative infinity
    for i in range(seq_length):
        for j in range(seq_length):
            if i >= seq_length or j >= seq_length:
                tl.store(output_ptr + offset + i * seq_len + j, float('-inf'))

import torch
import triton

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['batch_size', 'num_heads', 'seq_len']
)
@triton.jit
def token_softmax_fwd(
    logits: torch.Tensor,  # Input logits (shape: [batch_size, num_heads, seq_len, seq_len])
    seq_lengths: torch.Tensor,  # Sequence lengths (shape: [batch_size])
    output: torch.Tensor,  # Output softmax values (shape: [batch_size, num_heads, seq_len, seq_len])
    BLOCK_SIZE: tl.constexpr
):
    batch_size, num_heads, seq_len, _ = logits.shape

    # Launch the kernel
    grid = (batch_size * num_heads,)
    _fwd_kernel_token_softmax[grid](
        logits, output, seq_lengths, batch_size, num_heads, seq_len, BLOCK_SIZE
    )
