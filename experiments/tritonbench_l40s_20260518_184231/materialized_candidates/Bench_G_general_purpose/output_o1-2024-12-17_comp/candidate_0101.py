import triton
import triton.language as tl


HIDDEN_DIM = 1024      # Example hidden dimension
N_ELEMENTS = 128       # Example number of elements used internally
BLOCK_SIZE = 128       # Example block size
NUM_SEQS = 4           # Example number of sequences (decoding case)


@triton.jit
def prefill_cache_kernel(
    cos_cache_ptr, sin_cache_ptr,
    cos_output_ptr, sin_output_ptr,
    cumsum_lengths_ptr, offset_ptr,
    total_length, hidden_dim,
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Each program handles one block of data (BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_length

    # Load offsets from cumsum_lengths for index computations
    # Note: offset_ptr is the base offset for cos_cache/sin_cache
    offset_val = tl.load(offset_ptr)
    cum_val = tl.load(cumsum_lengths_ptr + offsets, mask=mask, other=0)

    # Compute the index into cos_cache/sin_cache
    cos_src_idx = (cum_val + offset_val) * hidden_dim
    sin_src_idx = (cum_val + offset_val) * hidden_dim

    # Destination index for cos_output/sin_output
    cos_dst_idx = offsets * hidden_dim
    sin_dst_idx = offsets * hidden_dim

    # Loop over hidden dimension in increments of n_elements
    for i in range(0, hidden_dim, n_elements):
        gather_idx = tl.arange(0, n_elements) + i
        in_bounds = gather_idx < hidden_dim
        cos_val = tl.load(
            cos_cache_ptr + (cos_src_idx + gather_idx),
            mask=mask & in_bounds,
            other=0.0
        )
        sin_val = tl.load(
            sin_cache_ptr + (sin_src_idx + gather_idx),
            mask=mask & in_bounds,
            other=0.0
        )

        tl.store(
            cos_output_ptr + (cos_dst_idx + gather_idx),
            cos_val,
            mask=mask & in_bounds
        )
        tl.store(
            sin_output_ptr + (sin_dst_idx + gather_idx),
            sin_val,
            mask=mask & in_bounds
        )


@triton.jit
def decoding_cache_kernel(
    cos_cache_ptr, sin_cache_ptr,
    cos_output_ptr, sin_output_ptr,
    lengths_ptr,
    hidden_dim,
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Each program handles one sequence
    seq_idx = pid
    seq_len = tl.load(lengths_ptr + seq_idx)
    decode_step = seq_len - 1  # point to previous entry in cache
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    # Base index in cache
    base_cache_idx = decode_step * hidden_dim

    # Load from cos_cache/sin_cache
    cos_val = tl.load(
        cos_cache_ptr + base_cache_idx + offsets,
        mask=mask,
        other=0.0
    )
    sin_val = tl.load(
        sin_cache_ptr + base_cache_idx + offsets,
        mask=mask,
        other=0.0
    )

    # Store into outputs
    out_base_idx = seq_idx * hidden_dim
    tl.store(cos_output_ptr + out_base_idx + offsets, cos_val, mask=mask)
    tl.store(sin_output_ptr + out_base_idx + offsets, sin_val, mask=mask)


def get_xine_cache(
    cos_cache, sin_cache,
    lengths, is_prompts,
    device='cuda'
):
    import torch

    if is_prompts:
        # Precompute cumsum of lengths
        cumsum_lengths = torch.cumsum(lengths, dim=0) - 1
        total_length = cumsum_lengths[-1].item() + 1

        cos_output = torch.empty(
            (total_length, HIDDEN_DIM),
            dtype=cos_cache.dtype,
            device=device
        )
        sin_output = torch.empty_like(cos_output)

        offsets = torch.tensor([0], dtype=torch.int32, device=device)
        grid = lambda meta: ( (total_length + BLOCK_SIZE - 1) // BLOCK_SIZE, )
        prefill_cache_kernel[grid](
            cos_cache, sin_cache,
            cos_output, sin_output,
            cumsum_lengths, offsets,
            total_length, HIDDEN_DIM,
            N_ELEMENTS,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return cos_output, sin_output
    else:
        num_seqs = lengths.shape[0]
        cos_output = torch.empty(
            (num_seqs, HIDDEN_DIM),
            dtype=cos_cache.dtype,
            device=device
        )
        sin_output = torch.empty_like(cos_output)

        grid = lambda meta: (num_seqs,)
        decoding_cache_kernel[grid](
            cos_cache, sin_cache,
            cos_output, sin_output,
            lengths,
            HIDDEN_DIM,
            N_ELEMENTS,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return cos_output, sin_output
