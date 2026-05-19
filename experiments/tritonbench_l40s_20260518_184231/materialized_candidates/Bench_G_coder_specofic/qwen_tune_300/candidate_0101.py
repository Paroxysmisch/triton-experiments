import torch
import triton
import triton.language as tl

@triton.jit
def prefill_cache_kernel(
    cos_cache, sin_cache,
    cos_output, sin_output,
    cumsum_lengths,
    HIDDEN_DIM: tl.constexpr,
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N_ELEMENTS

    seq_idx = offset // HIDDEN_DIM
    is_valid_seq = seq_idx < cumsum_lengths.shape[0]
    seq_len = tl.load(cumsum_lengths + seq_idx, mask=mask & is_valid_seq, other=0)
    prev_seq_len = tl.where(seq_idx > 0, tl.load(cumsum_lengths + seq_idx - 1, mask=mask & is_valid_seq, other=0), 0)
    cos_mask = (offset - prev_seq_len < seq_len) & (offset - prev_seq_len >= 0)
    sin_mask = (offset - prev_seq_len < seq_len) & (offset - prev_seq_len >= 0)

    cos_cache_idx = tl.load(cos_cache + offset, mask=cos_mask, other=0.0)
    sin_cache_idx = tl.load(sin_cache + offset, mask=sin_mask, other=0.0)

    tl.store(cos_output + offset, cos_cache_idx, mask=mask)
    tl.store(sin_output + offset, sin_cache_idx, mask=mask)

@triton.jit
def decoding_cache_kernel(
    cos_cache, sin_cache,
    cos_output, sin_output,
    lengths,
    HIDDEN_DIM: tl.constexpr,
    NUM_SEQS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE) + seq_idx * HIDDEN_DIM
    mask = offset < NUM_SEQS * HIDDEN_DIM

    length_1 = tl.load(lengths + seq_idx)
    prev_length_1 = tl.where(seq_idx > 0, tl.load(lengths + seq_idx - 1), 0)
    length_mask = offset < length_1 * HIDDEN_DIM
    prev_length_mask = (offset >= prev_length_1 * HIDDEN_DIM) & (offset < length_1 * HIDDEN_DIM)

    cos_cache_idx = tl.load(cos_cache + offset, mask=prev_length_mask, other=0.0)
    sin_cache_idx = tl.load(sin_cache + offset, mask=prev_length_mask, other=0.0)

    tl.store(cos_output + offset, cos_cache_idx, mask=length_mask)
    tl.store(sin_output + offset, sin_cache_idx, mask=length_mask)

def get_xine_cache(cos_cache, sin_cache, lengths, is_prompts=False):
    if is_prompts:
        total_length = lengths.sum().item()
        cumsum_lengths = torch.cumsum(lengths, dim=0)
        cos_cache = cos_cache.contiguous()
        sin_cache = sin_cache.contiguous()
        cos_output = torch.empty(total_length, device=cos_cache.device, dtype=cos_cache.dtype)
        sin_output = torch.empty(total_length, device=sin_cache.device, dtype=sin_cache.dtype)

        BLOCK_SIZE = triton.next_power_of_2(int(cos_cache.shape[-1] / 4))
        grid = (triton.cdiv(total_length, BLOCK_SIZE),)
        prefill_cache_kernel[grid](
            cos_cache, sin_cache,
            cos_output, sin_output,
            cumsum_lengths,
            HIDDEN_DIM=cos_cache.shape[-1],
            N_ELEMENTS=total_length,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return cos_output, sin_output
    else:
        cos_cache = cos_cache.contiguous()
        sin_cache = sin_cache.contiguous()
        cos_output = torch.empty_like(cos_cache, device=cos_cache.device, dtype=cos_cache.dtype)
        sin_output = torch.empty_like(sin_cache, device=sin_cache.device, dtype=sin_cache.dtype)

        NUM_SEQS = lengths.shape[0]
        BLOCK_SIZE = triton.next_power_of_2(int(cos_cache.shape[-1] / 4))
        grid = (triton.cdiv(cos_cache.shape[0], BLOCK_SIZE), NUM_SEQS)
        decoding_cache_kernel[grid](
            cos_cache, sin_cache,
            cos_output, sin_output,
            lengths,
            HIDDEN_DIM=cos_cache.shape[-1],
            NUM_SEQS=NUM_SEQS,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return cos_output, sin_output
