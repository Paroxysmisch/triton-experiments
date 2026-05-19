import triton
import triton.language as tl

# Constants
HIDDEN_DIM = 1024
N_ELEMENTS = 128
NUM_SEQS = 64
BLOCK_SIZE = 128

@triton.jit
def prefill_cache_kernel(
    cos_cache_ptr, sin_cache_ptr, cumsum_lengths_ptr, cos_output_ptr, sin_output_ptr,
    cache_stride: tl.constexpr, hidden_stride: tl.constexpr, total_length: tl.constexpr,
    HIDDEN_DIM: tl.constexpr, N_ELEMENTS: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_length

    # Compute the original sequence index for each element
    cumsum_lengths = tl.load(cumsum_lengths_ptr + offsets, mask=mask, other=0)
    seq_indices = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for i in range(BLOCK_SIZE):
        if mask[i]:
            seq_indices[i] = tl.sum(cumsum_lengths[:i] < offsets[i])

    # Compute the cache indices
    cache_indices = offsets - cumsum_lengths[seq_indices]

    # Load the cache data
    cos_cache = tl.load(cos_cache_ptr + seq_indices * cache_stride + cache_indices * hidden_stride, mask=mask, other=0)
    sin_cache = tl.load(sin_cache_ptr + seq_indices * cache_stride + cache_indices * hidden_stride, mask=mask, other=0)

    # Store the cache data in the output tensors
    tl.store(cos_output_ptr + offsets * hidden_stride, cos_cache, mask=mask)
    tl.store(sin_output_ptr + offsets * hidden_stride, sin_cache, mask=mask)

@triton.jit
def decoding_cache_kernel(
    cos_cache_ptr, sin_cache_ptr, lengths_ptr, cos_output_ptr, sin_output_ptr,
    cache_stride: tl.constexpr, hidden_stride: tl.constexpr,
    HIDDEN_DIM: tl.constexpr, NUM_SEQS: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUM_SEQS

    # Compute the original sequence index for each block
    lengths = tl.load(lengths_ptr + offsets, mask=mask, other=0)
    seq_indices = offsets

    # Compute the cache indices
    cache_indices = lengths - 1

    # Load the cache data
    cos_cache = tl.load(cos_cache_ptr + seq_indices * cache_stride + cache_indices * hidden_stride, mask=mask, other=0)
    sin_cache = tl.load(sin_cache_ptr + seq_indices * cache_stride + cache_indices * hidden_stride, mask=mask, other=0)

    # Store the cache data in the output tensors
    tl.store(cos_output_ptr + offsets * hidden_stride, cos_cache, mask=mask)
    tl.store(sin_output_ptr + offsets * hidden_stride, sin_cache, mask=mask)

import torch

def get_xine_cache(
    cos_cache, sin_cache, cumsum_lengths, lengths, is_prompts,
    total_length, num_seqs, hidden_dim, n_elements, num_seqs_decoding
):
    # Constants
    HIDDEN_DIM = hidden_dim
    N_ELEMENTS = n_elements
    NUM_SEQS = num_seqs
    BLOCK_SIZE = n_elements

    # Calculate strides
    cache_stride = cos_cache.shape[1]
    hidden_stride = cos_cache.shape[2]

    # Allocate output tensors
    cos_output = torch.empty((total_length, hidden_dim), device=cos_cache.device, dtype=cos_cache.dtype)
    sin_output = torch.empty((total_length, hidden_dim), device=sin_cache.device, dtype=sin_cache.dtype)

    if is_prompts:
        # Launch prefill_cache_kernel
        grid = (total_length // BLOCK_SIZE + 1,)
        prefill_cache_kernel[grid](
            cos_cache, sin_cache, cumsum_lengths, cos_output, sin_output,
            cache_stride, hidden_stride, total_length,
            HIDDEN_DIM, N_ELEMENTS, BLOCK_SIZE
        )
    else:
        # Launch decoding_cache_kernel
        grid = (num_seqs_decoding // BLOCK_SIZE + 1,)
        decoding_cache_kernel[grid](
            cos_cache, sin_cache, lengths, cos_output, sin_output,
            cache_stride, hidden_stride,
            HIDDEN_DIM, NUM_SEQS, BLOCK_SIZE
        )

    return cos_output, sin_output
