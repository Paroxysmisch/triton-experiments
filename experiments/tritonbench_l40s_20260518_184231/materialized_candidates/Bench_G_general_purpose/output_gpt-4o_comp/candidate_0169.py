import triton
import triton.language as tl

# Define constants for head dimension and cache block size
HEAD_DIM = 64
KCACHE_X = 128

@triton.jit
def _copy_to_kcache_seqlen_n_kernel(
    k_ptr, cache_ptr, seqlen_ptr, 
    stride_k, stride_cache, 
    B, H, N_CTX, 
    layout_flag, BLOCK_SIZE: tl.constexpr
):
    # Get the program ID for the current grid
    pid = tl.program_id(axis=0)

    # Compute batch and head indices
    batch_idx = pid // H
    head_idx = pid % H

    # Calculate current token index
    token_idx = tl.arange(0, BLOCK_SIZE)

    # Compute the past sequence length for the current batch
    past_seqlen = tl.load(seqlen_ptr + batch_idx)

    # Compute the cache index based on the layout flag
    if layout_flag == 0:
        # Traditional layout
        cache_idx = past_seqlen + token_idx
    else:
        # New layout
        cache_idx = token_idx * KCACHE_X + past_seqlen

    # Calculate the input and output pointers
    k_offset = batch_idx * stride_k + head_idx * HEAD_DIM + token_idx
    cache_offset = batch_idx * stride_cache + head_idx * HEAD_DIM + cache_idx

    # Load the key vector from input
    k_data = tl.load(k_ptr + k_offset, mask=token_idx < N_CTX)

    # Store the key vector into the cache
    tl.store(cache_ptr + cache_offset, k_data, mask=token_idx < N_CTX)


def copy_k_to_blocked_cache(k, cache, seqlen, batch_size, head_size, context_size, layout_flag):
    # Ensure the input dimensions are compatible
    assert k.shape[0] == batch_size * head_size
    assert k.shape[1] == context_size

    # Calculate strides
    stride_k = k.stride(0)
    stride_cache = cache.stride(0)

    # Launch the kernel
    grid = (batch_size * head_size,)
    _copy_to_kcache_seqlen_n_kernel[grid](
        k_ptr=k, 
        cache_ptr=cache, 
        seqlen_ptr=seqlen, 
        stride_k=stride_k, 
        stride_cache=stride_cache, 
        B=batch_size, 
        H=head_size, 
        N_CTX=context_size, 
        layout_flag=layout_flag, 
        BLOCK_SIZE=HEAD_DIM
    )
