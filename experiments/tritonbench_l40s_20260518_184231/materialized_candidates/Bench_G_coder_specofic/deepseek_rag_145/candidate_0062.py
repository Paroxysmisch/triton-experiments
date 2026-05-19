@triton.jit
def embedding_kernel(
    weight_ptr,
    BLOCK_TD: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    seq_stride_ptr,
    token_ids_ptr,
    output_ptr,
    output_stride_ptr,
    mask_ptr
):
    # Load token IDs from memory
    token_ids = tl.load(token_ids_ptr)

    # Load seqence stride from memory
    seq_stride = tl.load(seq_stride_ptr)[0]

    # Compute the offset in the weight matrix for the current sequence
    offsets = token_ids * BLOCK_DMODEL * tl.float32

    # Load the mask from memory
    mask = tl.load(mask_ptr)

    # Initialize the output tensor in shared memory
    output = tl.zeros((BLOCK_SEQ, BLOCK_DMODEL), dtype=tl.float32)

    # Iterate over the dimensions of the weight matrix
    for d in range(BLOCK_DMODEL):
        # Compute the weight matrix address for the current dimension
        weight_addr = offsets + d

        # Load the weights from memory
        weights = tl.load(weight_ptr + weight_addr)

        # If the mask is set, apply the mask to the weights
        weights = weights & mask

        # Store the weights in the output tensor
        tl.store(output_ptr + d * seq_stride, output)
