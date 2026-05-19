import triton.lang as tl

@triton.jit
def _copy_to_kcache_seqlen_n_kernel(
    # input parameters
    tokens, seqlens, cache, offsets,
    # other parameters
    BLOCK_SIZE, HEAD_DIM, KCACHE_X,
    # output parameter
    output,
    # other parameters
    stride_x, stride_y, stride_z,
):
    # calculation of position within a blocked cache layout
    x = tl.program_id(axis=0)
    y = tl.program_id(axis=1)
    z = tl.program_id(axis=2)

    # current token and sequence indices
    token_idx = x * BLOCK_SIZE + y
    seq_idx = z * KCACHE_X + x

    # past sequence lengths
    past_seq_len = seqlens[seq_idx]

    # exact location in a cache where the token data should be stored
    token_offset = past_seq_len + offsets[seq_idx]

    # load data from tokens
    token_data = tl.load(tokens + token_offset, mask=token_idx < seqlens[seq_idx])

    # store data into cache
    tl.store(cache + token_offset, token_data, mask=token_idx < seqlens[seq_idx])
