import torch
import triton
import triton.language as tl

@triton.jit
def prefill_cache_kernel(
    cos_cache_ptr,
    sin_cache_ptr,
    cos_output_ptr,
    sin_output_ptr,
    cumsum_lengths_ptr,
    total_length,
    HIDDEN_DIM: tl.constexpr,
    cumsum_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_hidden = tl.program_id(1)
    
    block_start = pid_hidden * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < HIDDEN_DIM
    
    i = pid
    
    low = 0
    high = cumsum_len
    while low < high:
        mid = (low + high) // 2
        mid_val = tl.load(cumsum_lengths_ptr + mid)
        if mid_val <= i:
            low = mid + 1
        else:
            high = mid
    b = low - 1
    
    if b == 0:
        pos = i
    else:
        cumsum_prev = tl.load(cumsum_lengths_ptr + (b - 1))
        pos = i - cumsum_prev
    
    cos_cache_offset = pos * HIDDEN_DIM + offsets
    sin_cache_offset = pos * HIDDEN_DIM + offsets
    output_offset = i * HIDDEN_DIM + offsets
    
    cos_val = tl.load(cos_cache_ptr + cos_cache_offset, mask=mask, other=0)
    sin_val = tl.load(sin_cache_ptr + sin_cache_offset, mask=mask, other=0)
    
    tl.store(cos_output_ptr + output_offset, cos_val, mask=mask)
    tl.store(sin_output_ptr + output_offset, sin_val, mask=mask)

@triton.jit
def decoding_cache_kernel(
    cos_cache_ptr,
    sin_cache_ptr,
    lengths_ptr,
    cos_output_ptr,
    sin_output_ptr,
    NUM_SEQS: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_hidden = tl.program_id(1)
    
    block_start = pid_hidden * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < HIDDEN_DIM
    
    s = pid
    length = tl.load(lengths_ptr + s)
    pos = length - 1
    
    cos_cache_offset = pos * HIDDEN_DIM + offsets
    sin_cache_offset = pos * HIDDEN_DIM + offsets
    output_offset = s * HIDDEN_DIM + offsets
    
    cos_val = tl.load(cos_cache_ptr + cos_cache_offset, mask=mask, other=0)
    sin_val = tl.load(sin_cache_ptr + sin_cache_offset, mask=mask, other=0)
    
    tl.store(cos_output_ptr + output_offset, cos_val, mask=mask)
    tl.store(sin_output_ptr + output_offset, sin_val, mask=mask)

def get_xine_cache(
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    lengths: torch.Tensor,
    cumsum_lengths: torch.Tensor,
    is_prompts: bool,
):
    assert cos_cache.is_cuda and sin_cache.is_cuda
    assert lengths.is_cuda
    HIDDEN_DIM = cos_cache.size(-1)
    BLOCK_SIZE = 128
    
    if is_prompts:
        assert cumsum_lengths is not None
        assert cumsum_lengths.is_cuda
        total_length = cumsum_lengths[-1].item()
        cumsum_len = cumsum_lengths.size(0)
        cos_output = torch.empty((total_length, HIDDEN_DIM), device=cos_cache.device, dtype=cos_cache.dtype)
        sin_output = torch.empty_like(cos_output)
        grid = (total_length, triton.cdiv(HIDDEN_DIM, BLOCK_SIZE))
        prefill_cache_kernel[grid](
            cos_cache, sin_cache, cos_output, sin_output, cumsum_lengths,
            total_length, HIDDEN_DIM, cumsum_len, BLOCK_SIZE
        )
    else:
        NUM_SEQS = lengths.size(0)
        cos_output = torch.empty((NUM_SEQS, HIDDEN_DIM), device=cos_cache.device, dtype=cos_cache.dtype)
        sin_output = torch.empty_like(cos_output)
        grid = (NUM_SEQS, triton.cdiv(HIDDEN_DIM, BLOCK_SIZE))
        decoding_cache_kernel[grid](
            cos_cache, sin_cache, lengths, cos_output, sin_output,
            NUM_SEQS, HIDDEN_DIM, BLOCK_SIZE
        )
    return cos_output, sin_output
