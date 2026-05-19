+ tl.arange(0, HIDDEN_DIM)[None, :] * hidden_stride),
        sin_cache_part,
        mask=idx[:, None] < NUM_SEQS,
    )


def get_xine_cache(cos_cache, sin_cache, lengths, is_prompts, device):
    assert cos_cache.shape[-1] == sin_cache.shape[-1]
    assert cos_cache.is_contiguous() and sin_cache.is_contiguous()
    hidden_dim = cos_cache.shape[-1]
    num_seqs = lengths.numel()
    block_size = 512

    if is_prompts:
        total_length = lengths.sum().item()
        cumsum_lengths = torch.cumsum(lengths, dim=0, dtype=torch.long).to(device)
        cos_output = torch.empty(total_length, hidden_dim, device=device, dtype=cos_cache.dtype)
        sin_output = torch.empty(total_length, hidden_dim, device=device, dtype=cos_cache.dtype)
        grid = lambda meta: (
            triton.cdiv(total_length, meta["BLOCK_SIZE"]),
            triton.cdiv(hidden_dim, meta["N_ELEMENTS"]),
        )
        prefill_cache_kernel[grid](
            cos_cache,
            sin_cache,
            cumsum_lengths,
            cos_output,
            sin_output,
            cos_cache.stride(0),
            cos_cache.stride(1),
            total_length,
            HIDDEN_DIM=hidden_dim,
            N_ELEMENTS=triton.next_power_of_2(hidden_dim),
            BLOCK_SIZE=block_size,
        )
        return cos_output, sin_output
    else:
        cos_output = torch.empty(num_seqs, hidden_dim, device=device, dtype=cos_cache.dtype)
        sin_output = torch.empty(num_seqs, hidden_dim, device=device, dtype=cos_cache.dtype)
        grid = lambda meta: (triton.cdiv(num_seqs, meta["BLOCK_SIZE"]), 1)
        decoding_cache_kernel[grid](
            cos_cache,
            sin_cache,
            lengths,
            cos_output,
            sin_output,
            cos_cache.stride(0),
            cos_cache.stride(1),
            HIDDEN_DIM=hidden_dim,
            NUM_SEQS=num_seqs,
            BLOCK_SIZE=block_size,
        )
        return cos_output, sin_output
