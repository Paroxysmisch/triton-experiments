import triton
import triton.language as tl


@triton.jit
def embedding_kernel(
    token_ids_ptr,  # pointer to token IDs
    weight_ptr,     # pointer to weight matrix
    out_ptr,        # pointer to output tensor
    seq_len,        # total number of tokens in sequence
    d_model,        # embedding dimension
    stride_w,       # stride for weight
    stride_out,     # stride for output
    BLOCK_N: tl.constexpr,    # block size for sequence dimension
    BLOCK_DMODEL: tl.constexpr  # block size for embedding dimension
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    # Offsets within the sequence
    r_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Offsets within the embedding dimension
    c_offsets = pid_d * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL)

    # Masks to guard memory operations
    mask_r = r_offsets < seq_len
    mask_c = c_offsets < d_model

    # Gather token IDs
    token_ids = tl.load(
        token_ids_ptr + r_offsets,
        mask=mask_r,
        other=0
    )

    # Compute read offsets in weight matrix:
    # weight is assumed to be [vocab_size, d_model],
    # so index = token_ids * stride_w + column
    read_offsets = token_ids * stride_w + c_offsets

    # Load embeddings
    embeddings = tl.load(
        weight_ptr + read_offsets,
        mask=mask_r & mask_c,
        other=0
    )

    # Compute write offsets in output:
    # out is assumed to be [seq_len, d_model],
    # so index = row * stride_out + column
    write_offsets = r_offsets * stride_out + c_offsets

    # Store embeddings
    tl.store(
        out_ptr + write_offsets,
        embeddings,
        mask=mask_r & mask_c
    )


def next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length()


def embedding(token_ids, weight, out):
    """
    token_ids: 1-D tensor of shape [seq_len]
    weight: 2-D tensor of shape [vocab_size, d_model]
    out: 2-D tensor of shape [seq_len, d_model]
    """
    seq_len = token_ids.shape[0]
    vocab_size, d_model = weight.shape

    # Compute strides
    stride_w = weight.stride(0)
    stride_out = out.stride(0)

    # Define block sizes
    BLOCK_N = 128
    BLOCK_DMODEL = next_power_of_two(d_model)

    # Grid to cover the sequence and embedding dimension
    grid_n = (seq_len + BLOCK_N - 1) // BLOCK_N
    grid_d = (d_model + BLOCK_DMODEL - 1) // BLOCK_DMODEL

    # Launch the kernel
    embedding_kernel[grid_n, grid_d](
        token_ids,
        weight,
        out,
        seq_len,
        d_model,
        stride_w,
        stride_out,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=2
    )
