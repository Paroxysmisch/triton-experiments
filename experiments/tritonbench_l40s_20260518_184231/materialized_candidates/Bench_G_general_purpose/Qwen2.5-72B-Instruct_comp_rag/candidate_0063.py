import triton
import triton.language as tl
import torch

@triton.jit
def embedding_kernel(
    tokens_ptr,  # Pointer to the token IDs
    tokens_batch_stride,  # Stride for the batch dimension in tokens
    tokens_seq_stride,  # Stride for the sequence dimension in tokens
    tokens_n_stride,  # Stride for the token dimension in tokens
    weights_ptr,  # Pointer to the weight matrix
    weights_batch_stride,  # Stride for the batch dimension in weights
    weights_dim_stride,  # Stride for the dimension in weights
    weights_n_stride,  # Stride for the token dimension in weights
    out_ptr,  # Pointer to the output tensor
    out_batch_stride,  # Stride for the batch dimension in out
    out_seq_stride,  # Stride for the sequence dimension in out
    out_dim_stride,  # Stride for the dimension in out
    n_tokens: tl.constexpr,  # Number of tokens
    n_dim: tl.constexpr,  # Dimension of the embedding vectors
    BLOCK_N: tl.constexpr,  # Block size for the sequence dimension
    BLOCK_NN: tl.constexpr,  # Block size for the token dimension
    BLOCK_DMODEL: tl.constexpr  # Block size for the dimension
):
    # Compute the offsets for the current program
    pid = tl.program_id(axis=0)
    batch_id = pid // (n_tokens // BLOCK_N)
    seq_id = (pid % (n_tokens // BLOCK_N)) * BLOCK_N

    # Iterate over the sequence dimension
    for seq_offset in range(seq_id, min(seq_id + BLOCK_N, n_tokens), BLOCK_NN):
        # Load the token ID
        token_id = tl.load(tokens_ptr + batch_id * tokens_batch_stride + seq_offset * tokens_n_stride)

        # Iterate over the dimension
        for dim_offset in range(0, n_dim, BLOCK_DMODEL):
            # Load the embedding vector
            weight_vec = tl.load(weights_ptr + batch_id * weights_batch_stride + token_id * weights_n_stride + dim_offset * weights_dim_stride, mask=dim_offset + tl.arange(0, BLOCK_DMODEL) < n_dim)

            # Store the embedding vector in the output tensor
            tl.store(out_ptr + batch_id * out_batch_stride + seq_offset * out_seq_stride + dim_offset * out_dim_stride, weight_vec, mask=dim_offset + tl.arange(0, BLOCK_DMODEL) < n_dim)

def embedding(tokens: torch.Tensor, weights: torch.Tensor, out: Optional[torch.Tensor] = None):
    # Ensure the output tensor is allocated
    if out is None:
        out = torch.empty_like(tokens, dtype=weights.dtype, device=weights.device)

    # Get the dimensions
    n_batch, n_tokens = tokens.shape
    n_dim = weights.shape[1]

    # Determine the block sizes
    BLOCK_N = 128
    BLOCK_NN = 16
    BLOCK_DMODEL = 16

    # Align BLOCK_DMODEL to the next power of two based on weight dimensions
    BLOCK_DMODEL = 2 ** (BLOCK_DMODEL - 1).bit_length()

    # Launch the kernel
    grid = (n_batch * (n_tokens // BLOCK_N),)
    embedding_kernel[grid](
        tokens,  # Pointer to the token IDs
        tokens.stride(0),  # Stride for the batch dimension in tokens
        tokens.stride(1),  # Stride for the sequence dimension in tokens
        1,  # Stride for the token dimension in tokens
        weights,  # Pointer to the weight matrix
        weights.stride(0),  # Stride for the batch dimension in weights
        weights.stride(1),  # Stride for the dimension in weights
        1,  # Stride for the token dimension in weights
        out,  # Pointer to the output tensor
        out.stride(0),  # Stride for the batch dimension in out
        out.stride(1),  # Stride for the sequence dimension in out
        out.stride(2),  # Stride for the dimension in out
        n_tokens,  # Number of tokens
        n_dim,  # Dimension of the embedding vectors
        BLOCK_N,  # Block size for the sequence dimension
        BLOCK_NN,  # Block size for the token dimension
        BLOCK_DMODEL  # Block size for the dimension
    )

    return out
