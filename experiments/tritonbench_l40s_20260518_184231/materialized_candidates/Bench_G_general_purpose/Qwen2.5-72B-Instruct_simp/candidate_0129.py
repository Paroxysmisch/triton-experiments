import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_apply_penalty(
    logits_ptr,  # Pointer to the logits tensor
    presence_penalty_ptr,  # Pointer to the presence penalty tensor
    frequency_penalty_ptr,  # Pointer to the frequency penalty tensor
    repetition_penalty_ptr,  # Pointer to the repetition penalty tensor
    token_ids_ptr,  # Pointer to the token IDs tensor
    token_counts_ptr,  # Pointer to the token counts tensor
    sequence_lengths_ptr,  # Pointer to the sequence lengths tensor
    batch_size,  # Number of batches
    vocab_size,  # Vocabulary size
    BLOCK_SIZE: tl.constexpr  # Block size
):
    # Get the batch index
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return

    # Load the sequence length for this batch
    seq_len = tl.load(sequence_lengths_ptr + batch_idx)

    # Load the penalties for this batch
    presence_penalty = tl.load(presence_penalty_ptr + batch_idx)
    frequency_penalty = tl.load(frequency_penalty_ptr + batch_idx)
    repetition_penalty = tl.load(repetition_penalty_ptr + batch_idx)

    # Compute the range of tokens for this batch
    token_start = batch_idx * seq_len
    token_end = token_start + seq_len

    # Iterate over the tokens in this batch
    for token_idx in range(token_start, token_end, BLOCK_SIZE):
        # Compute the block of tokens to process
        block_start = token_idx
        block_end = min(token_idx + BLOCK_SIZE, token_end)

        # Load the token IDs and token counts for this block
        token_ids = tl.load(token_ids_ptr + block_start, mask=block_start < block_end, other=0)
        token_counts = tl.load(token_counts_ptr + block_start, mask=block_start < block_end, other=0)

        # Compute the penalties for each token in the block
        for i in range(block_end - block_start):
            token_id = token_ids[i]
            token_count = token_counts[i]

            # Apply presence penalty
            if token_count > 0:
                logits_ptr[batch_idx * vocab_size + token_id] -= presence_penalty

            # Apply frequency penalty
            logits_ptr[batch_idx * vocab_size + token_id] -= frequency_penalty * token_count

            # Apply repetition penalty
            if token_count > 0:
                logits_ptr[batch_idx * vocab_size + token_id] /= repetition_penalty

    # Synchronize to ensure all penalties are applied
    tl.device_barrier()

import torch

def apply_penalty(
    logits: torch.Tensor,
    presence_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    repetition_penalty: torch.Tensor,
    token_ids: torch.Tensor,
    token_counts: torch.Tensor,
    sequence_lengths: torch.Tensor
):
    # Ensure all tensors are on the same device
    device = logits.device
    assert all(t.device == device for t in [presence_penalty, frequency_penalty, repetition_penalty, token_ids, token_counts, sequence_lengths])

    # Get the batch size and vocabulary size
    batch_size = logits.size(0)
    vocab_size = logits.size(1)

    # Calculate the block size based on the maximum sequence length in the batch
    max_seq_len = sequence_lengths.max().item()
    block_size = 128  # Adjust this based on your GPU and problem size

    # Configure the grid and block dimensions
    grid = (batch_size, 1, 1)
    block = (block_size, 1, 1)

    # Launch the kernel
    _fwd_kernel_apply_penalty[grid, block](
        logits.data_ptr(),
        presence_penalty.data_ptr(),
        frequency_penalty.data_ptr(),
        repetition_penalty.data_ptr(),
        token_ids.data_ptr(),
        token_counts.data_ptr(),
        sequence_lengths.data_ptr(),
        batch_size,
        vocab_size,
        block_size
    )

    return logits
