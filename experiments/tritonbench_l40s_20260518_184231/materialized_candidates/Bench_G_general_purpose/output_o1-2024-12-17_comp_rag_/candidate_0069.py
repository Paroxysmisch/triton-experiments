import triton
import triton.language as tl
import torch

@triton.jit
def embedding_kernel(
    weight_ptr,            # [vocab_size, d_model]
    token_ids_ptr,         # [seq_len]
    out_ptr,               # [seq_len, d_model]
    seq_len,               # scalar
    vocab_size,            # scalar
    d_model,               # scalar
    stride_w,              # stride for weight: typically d_model
    stride_o,              # stride for output: typically d_model
    BLOCK_N: tl.constexpr, # tile size for token dimension
    BLOCK_DMODEL: tl.constexpr # tile size for embedding dimension
):
    # Program IDs
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    # Ranges for token IDs and embedding dims
    token_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offsets = pid_d * BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL)

    # Create masks for valid tokens and dims
    mask_tokens = token_offsets < seq_len
    mask_d = d_offsets < d_model

    # Load token IDs
    token_ids = tl.load(token_ids_ptr + token_offsets, mask=mask_tokens, other=0)

    # For each token in the block, load embeddings
    # Each token ID corresponds to a row in weight
    # The column in weight is given by d_offsets
    # Embedding = weight[token_id, d_offsets]
    # So weight index = token_id * stride_w + d_offsets
    # We'll gather for each token index
    emb = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    for i in range(BLOCK_N):
        token_id = token_ids[i]
        is_valid = mask_tokens[i]
        row_offset = token_id * stride_w
        col_offset = d_offsets
        mask_load = mask_d & is_valid & (token_id < vocab_size)
        emb[i] = tl.load(weight_ptr + row_offset + col_offset, mask=mask_load, other=0.0)

    # Store embeddings to out
    # out index = token_idx * stride_o + d_offsets
    # token_idx is token_offsets
    for i in range(BLOCK_N):
        is_valid = mask_tokens[i]
        out_offset = token_offsets[i] * stride_o
        mask_store = mask_d & is_valid
        tl.store(out_ptr + out_offset + d_offsets, emb[i], mask=mask_store)


def embedding(weight: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """
    weight: [vocab_size, d_model], float32
    token_ids: [seq_len], int64
    returns out: [seq_len, d_model], float32
    """
    seq_len = token_ids.shape[0]
    vocab_size, d_model = weight.shape
    out = torch.empty((seq_len, d_model), device=weight.device, dtype=weight.dtype)

    # Compute block sizes
    BLOCK_N = 128  # tile size for token dimension
    # Round embedding dimension up to next power of 2
    # Ensures better performance for many GPU kernels
    next_pow2 = 1 << (d_model - 1).bit_length()
    BLOCK_DMODEL = min(next_pow2, 1024)

    grid = (
        ( (seq_len + BLOCK_N - 1) // BLOCK_N ), 
        ( (d_model + BLOCK_DMODEL - 1) // BLOCK_DMODEL )
    )

    # Strides
    stride_w = weight.stride(0)  # typically d_model
    stride_o = out.stride(0)     # typically d_model

    triton.run(
        kernel=embedding_kernel,
        grid=grid,
        args=[
            weight, 
            token_ids, 
            out, 
            seq_len, 
            vocab_size, 
            d_model, 
            stride_w, 
            stride_o
        ],
        num_warps=4,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL
    )

    return out
