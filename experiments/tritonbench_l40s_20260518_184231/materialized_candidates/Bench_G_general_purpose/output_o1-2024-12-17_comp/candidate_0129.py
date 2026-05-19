import triton
import triton.language as tl
from triton.runtime.autotuner import next_power_of_2

@triton.jit
def _fwd_kernel_apply_penalty(
    logits_ptr,            # *f32
    presence_ptr,          # *f32
    frequency_ptr,         # *f32
    repetition_ptr,        # *f32
    token_ids_ptr,         # *int32
    token_counts_ptr,      # *int32
    cumsum_seq_len_ptr,    # *int32
    stride,                # i32
    cur_batch,             # i32
    BLOCK: tl.constexpr
):
    # Load the penalty coefficients for this batch
    presence = tl.load(presence_ptr + cur_batch)
    frequency = tl.load(frequency_ptr + cur_batch)
    repetition = tl.load(repetition_ptr + cur_batch)

    # Determine the start and end range of token indices for this batch
    start_offset = tl.load(cumsum_seq_len_ptr + cur_batch)
    end_offset = tl.load(cumsum_seq_len_ptr + cur_batch + 1)

    # Compute the indices of the tokens to process
    offsets = start_offset + tl.arange(0, BLOCK)
    mask = offsets < end_offset

    # Load the token IDs and counts for these offsets
    token_ids = tl.load(token_ids_ptr + offsets, mask=mask, other=0)
    counts = tl.load(token_counts_ptr + offsets, mask=mask, other=0)

    # Base pointer for accessing logits
    base_ptr = logits_ptr + cur_batch * stride

    # Load the logits for the relevant token positions
    logits_val = tl.load(base_ptr + token_ids, mask=mask, other=0.0)

    # Apply repetition penalty
    logits_val = logits_val / repetition

    # Apply frequency penalty
    logits_val = logits_val - (frequency * counts)

    # Apply presence penalty if token appeared at least once
    presence_mask = counts > 0
    logits_val = tl.where(presence_mask, logits_val - presence, logits_val)

    # Store the updated logits
    tl.store(base_ptr + token_ids, logits_val, mask=mask)

def apply_penalty(
    Logits,
    presence_penalties,
    frequency_penalties,
    repetition_penalties,
    token_ids,
    token_counts,
    cumsum_seq_len
):
    # Ensure Logits is contiguous
    if not Logits.is_contiguous():
        Logits = Logits.contiguous()

    # Determine the maximum number of tokens in any single batch
    B, _ = Logits.shape
    max_len = 0
    for i in range(B):
        length = cumsum_seq_len[i + 1] - cumsum_seq_len[i]
        if length > max_len:
            max_len = length

    # Compute an efficient block size
    BLOCK = max(128, next_power_of_2(max_len))
    stride = Logits.stride(0)

    # Launch the kernel for each batch
    for i in range(B):
        _fwd_kernel_apply_penalty[(1,)](
            Logits,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
            token_ids,
            token_counts,
            cumsum_seq_len,
            stride,
            i,
            BLOCK,
            num_warps=8
        )
    return Logits
