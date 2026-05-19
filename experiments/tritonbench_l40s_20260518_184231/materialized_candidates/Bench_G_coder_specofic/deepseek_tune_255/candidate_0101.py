import torch
import triton
import triton.language as tl

@triton.jit
def prefill_cache_kernel(
    cos_cache,
    sin_cache,
    cumsum_lengths,
    cos_output,
    sin_output,
    HIDDEN_DIM: tl.constexpr,
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    cumsum_lengths_ptr = cumsum_lengths + pid
    target_index = tl.arange(0, BLOCK_SIZE)
    source_index = target_index % N_ELEMENTS
    original_seq_index = tl.cumsum(cumsum_lengths_ptr) - 1
    source_seq_index = original_seq_index - source_index
    mask = source_seq_index >= 0
    cache_cos = tl.load(cos_cache + source_seq_index * HIDDEN_DIM + source_index, mask=mask)
    cache_sin = tl.load(sin_cache + source_seq_index * HIDDEN_DIM + source_index, mask=mask)
    tl.store(cos_output + target_index, cache_cos, mask=mask)
    tl.store(sin_output + target_index, cache_sin, mask=mask)

@triton.jit
def decoding_cache_kernel(
    cos_cache,
    sin_cache,
    lengths,
    cos_output,
    sin_output,
    HIDDEN_DIM: tl.constexpr,
    NUM_SEQS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    seq_index = tl.program_id(0)
    length = tl.load(lengths + seq_index) - 1
    target_index = tl.arange(0, BLOCK_SIZE)
    source_index = target_index + length
    mask = source_index >= 0
    cache_cos = tl.load(cos_cache + source_index * HIDDEN_DIM + target_index, mask=mask)
    cache_sin = tl.load(sin_cache + source_index * HIDDEN_DIM + target_index, mask=mask)
    tl.store(cos_output + target_index, cache_cos, mask=mask)
    tl.store(sin_output + target_index, cache_sin, mask=mask)

def get_xine_cache(
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    lengths: torch.Tensor,
    cumsum_lengths: torch.Tensor,
    total_length: int,
    is_prompts: bool,
):
    HIDDEN_DIM = cos_cache.shape[-1]
    N_ELEMENTS = cos_cache.numel() // cos_cache.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(N_ELEMENTS)
    NUM_SEQS = lengths.shape[0]
    with torch.cuda.device(cos_cache.device):
        if is_prompts:
            output = torch.empty(
                NUM_SEQS,
                BLOCK_SIZE,
                HIDDEN_DIM,
                dtype=cos_cache.dtype,
                device=cos_cache.device,
            )
            grid = (total_length,)
            prefill_cache_kernel[grid](
                cos_cache,
                sin_cache,
                cumsum_lengths,
                output,
                output,
                HIDDEN_DIM,
                N_ELEMENTS,
                BLOCK_SIZE,
            )
        else:
            output = torch.empty(
                NUM_SEQS,
                BLOCK_SIZE,
                HIDDEN_DIM,
                dtype=cos_cache.dtype,
                device=cos_cache.device,
            )
            grid = (NUM_SEQS,)
            decoding_cache_kernel[grid](
                cos_cache,
                sin_cache,
                lengths,
                output,
                output,
                HIDDEN_DIM,
                NUM_SEQS,
                BLOCK_SIZE,
            )
    return output
