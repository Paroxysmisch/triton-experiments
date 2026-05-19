import triton
import triton.language as tl
import math
from torch._inductor.triton_heuristics import grid
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@triton.jit
def embedding_kernel(
    # The offsets determining which sequence to process
    pid,
    # The stride representing the size of each block of seqs
    BLOCK_N: tl.constexpr,
    # The stride representing the size of each block of nn
    BLOCK_NN: tl.constexpr,
    # The weight matrix containing the embeddings
    weight,
    # The sequence of token IDs
    token_ids,
    # A mask indicating valid positions within the sequences
    token_ids_mask,
    # The tensor where computed embeddings will be stored
    out,
    # The stride representing the size of the feature dimension in the out tensor
    OUT_BLOCK_DMODEL_STRIDE: tl.constexpr,
    # The number of embedding dimensions
    WEIGHT_DMODEL: tl.constexpr,
):
    # Compute the offsets for the current sequence block
    n_offsets = pid[0] * BLOCK_N + tl.arange(0, BLOCK_N)
    nn_offsets = pid[1] * BLOCK_NN + tl.arange(0, BLOCK_NN)

    # Load the current batch's token IDs
    token_ids = token_ids + n_offsets
    # Apply the mask to ensure we only access valid positions
    token_ids = tl.where(n_offsets < token_ids_mask, token_ids, 0)

    # Load the current batch's token IDs
    token_ids = tl.load(token_ids, mask=n_offsets < token_ids_mask)

    # Construct a 2D mask for the nn dimension
    # This helps us not go out of bounds when we address the weight tensor
    nn_mask = nn_offsets[:, None] < token_ids_mask

    # Load the embedding vectors, one by one for each token ID
    arange = tl.arange(0, WEIGHT_DMODEL)

    # Address the weight tensor using the current batch's token IDs
    # and the arange to create offsets for each dimension
    # This is achieved using tl.load with the mask applied to nn_mask
    weight = weight + (token_ids[:, None] * arange)
    weight = tl.load(weight, mask=nn_mask, other=0.0)
    
    # Store the loaded embeddings in the output tensor
    out = out + (n_offsets[:, None] * OUT_BLOCK_DMODEL_STRIDE) + arange
    tl.store(out, weight, mask=nn_mask)

@embedding_wrapper
def embedding(
    weight,
    token_ids,
    token_ids_mask,
    out,
    max_grid=None,
):
    if max_grid is None:
        # Compute a rough maximum grid size needed for all cases
        # We use `next_power_of_two` since the kernel uses `tl.load` hints
        # with access pointers passed manually which require a power of two.
        max_grid = (
            triton_helpers.next_power_of_two(token_ids_mask),
            1,
            1,
        )
    # Align BLOCK_DMODEL with the next power of two based on feature size
    ALIGN = 1
    weight_dmodel = weight.shape[1]
    triton_dmodel = xm.ntt.next_power_of_two(weight_dmodel)
    BLOCK_DMODEL = min(triton_dmodel, 4 * ALIGN * weight_dmodel)

    config = {
        "BLOCK_DMODEL": BLOCK_DMODEL,
        "BLOCK_N": 128,
        "BLOCK_NN": 128,
        "WEIGHT_DMODEL": [weight_dmodel],
        "GROUP_SIZE_N": 32,
    }

    instance = instance_descriptor(weight, token_ids, token_ids_mask, out)
    signature = triton_helpers.program_signature(config, dtypes=[weight.dtype])

    grid = (
        triton_helpers.cdiv(token_ids_mask, config["BLOCK_N"])[:, None],
        1,
        1,
    )

    return triton_helpers.launch(
        embedding_kernel, max_grid, config, instance, signature, grid=grid
    )﻿
