import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_apply_penalty(
    Logits, presence_penalty, frequency_penalty, repetition_penalty,
    p_token_ids, p_token_counts, p_cumsum_seq_len, 
    stride_logit_b, stride_logit_s,
    BLOCK_P: tl.constexpr
):
    # Determine the current batch index and load penalties
    cur_batch = tl.program_id(0)
    cur_frequency = tl.load(frequency_penalty + cur_batch)
    cur_presence = tl.load(presence_penalty + cur_batch)
    cur_repetition = tl.load(repetition_penalty + cur_batch)

    # Load the start and end indices for the current batch
    cur_batch_start_index = tl.load(p_cumsum_seq_len + cur_batch)
    cur_batch_end_index = tl.load(p_cumsum_seq_len + cur_batch + 1)

    # Compute the offsets and load token ids and their counts
    cur_batch_id_offset = cur_batch_start_index + tl.arange(0, BLOCK_P)
    batch_ids = tl.load(p_token_ids + cur_batch_id_offset, mask=cur_batch_id_offset < cur_batch_end_index, other=0)
    batch_ids_count = tl.load(p_token_counts + cur_batch_id_offset, mask=cur_batch_id_offset < cur_batch_end_index, other=0)
    
    # Compute the position in logits and adjust based on repetition, frequency, and presence penalties
    row_start_ptr = Logits + cur_batch * stride_logit_b
    cur_offset = row_start_ptr + batch_ids
    cur_logits = tl.load(cur_offset, mask=cur_batch_id_offset < cur_batch_end_index, other=0.0)
    
    # Apply repetition penalty
    rep_logits = tl.where(cur_logits > 0, cur_logits / cur_repetition, cur_logits * cur_repetition)
    
    # Apply frequency penalty
    freq_logits = rep_logits - batch_ids_count * cur_frequency
    
    # Apply presence penalty
    pre_logits = freq_logits - cur_presence
    
    # Store the adjusted logits
    output_ptr = Logits + cur_batch * stride_logit_b + batch_ids
    tl.store(output_ptr, pre_logits, mask=cur_batch_id_offset < cur_batch_end_index)

@torch.no_grad()
def apply_penalty(Logits, presence_penalty, frequency_penalty, repetition_penalty, p_token_ids, p_token_counts, p_cumsum_seq_len):
    assert Logits.is_contiguous()
    # Determine the appropriate BLOCK size based on the maximum sequence length
    p_max_len_in_batch = p_cumsum_seq_len[1:] - p_cumsum_seq_len[:-1]
    BLOCK = triton.next_power_of_2(p_max_len_in_batch.max().item())
    BLOCK = max(BLOCK, 512)
    num_warps = 8
    
    # Launch the Triton kernel with the determined configurations
    _fwd_kernel_apply_penalty[(Logits.shape[0], )](
        Logits, presence_penalty, frequency_penalty, repetition_penalty,
        p_token_ids, p_token_counts, p_cumsum_seq_len,
        Logits.stride(0), Logits.stride(1),
        num_warps=num_warps,
        BLOCK_P=BLOCK
    )
    return Logits
