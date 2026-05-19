import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(
    Q, K, V, O,  # Input and output tensors
    seq_lengths,  # Sequence lengths for each batch
    batch_size,   # Batch size
    head_dim,     # Head dimension
    num_heads,    # Number of heads
    BLOCK_SIZE_SEQ: tl.constexpr,  # Block size for sequences
    BLOCK_SIZE_HEAD: tl.constexpr  # Block size for heads
):
    # Get the current batch and head indices
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # Check if the current batch and head indices are within bounds
    if batch_idx >= batch_size or head_idx >= num_heads:
        return

    # Load the sequence length for the current batch
    seq_len = seq_lengths[batch_idx]

    # Initialize accumulators and logic variables
    acc = tl.zeros((BLOCK_SIZE_HEAD,), dtype=tl.float32)
    logic_sum = tl.zeros((BLOCK_SIZE_HEAD,), dtype=tl.float32)

    # Process each block of sequence data
    for start in range(0, seq_len, BLOCK_SIZE_SEQ):
        # Calculate the end index for the current block
        end = min(start + BLOCK_SIZE_SEQ, seq_len)

        # Load the sequence data for the current block
        q = tl.load(Q + batch_idx * head_dim * num_heads + head_idx * head_dim + start * BLOCK_SIZE_HEAD, mask=start + tl.arange(0, BLOCK_SIZE_SEQ) < seq_len)
        k = tl.load(K + batch_idx * head_dim * num_heads + head_idx * head_dim + start * BLOCK_SIZE_HEAD, mask=start + tl.arange(0, BLOCK_SIZE_SEQ) < seq_len)
        v = tl.load(V + batch_idx * head_dim * num_heads + head_idx * head_dim + start * BLOCK_SIZE_HEAD, mask=start + tl.arange(0, BLOCK_SIZE_SEQ) < seq_len)

        # Compute the scaling factor
        scale = 1.0 / tl.sqrt(tl.float32(head_dim))

        # Compute the attention scores
        attn_scores = tl.exp(q * k * scale)

        # Accumulate the attention scores and the value
        acc += attn_scores * v
        logic_sum += attn_scores

    # Normalize the accumulated value by the sum of exponentials
    acc /= logic_sum

    # Store the final output
    tl.store(O + batch_idx * head_dim * num_heads + head_idx * head_dim, acc, mask=tl.arange(0, BLOCK_SIZE_HEAD) < head_dim)

import torch

def flash_decode_stage2(Q, K, V, O, seq_lengths, batch_size, head_dim, num_heads, BLOCK_SIZE_SEQ=128, BLOCK_SIZE_HEAD=64):
    # Convert inputs to Triton tensors
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    O = O.contiguous()
    seq_lengths = seq_lengths.contiguous()

    # Determine the grid size
    grid = (batch_size, num_heads)

    # Launch the kernel
    _fwd_kernel_flash_decode_stage2[grid](
        Q, K, V, O,
        seq_lengths,
        batch_size,
        head_dim,
        num_heads,
        BLOCK_SIZE_SEQ,
        BLOCK_SIZE_HEAD
    )
