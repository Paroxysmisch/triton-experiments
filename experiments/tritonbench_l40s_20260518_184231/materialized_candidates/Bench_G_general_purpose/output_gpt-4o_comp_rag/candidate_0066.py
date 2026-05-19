import triton
import triton.language as tl
import torch

@triton.jit
def embedding_kernel(
    token_ids_ptr,  # Pointer to token IDs
    weight_ptr,     # Pointer to weight matrix
    out_ptr,        # Pointer to output tensor
    num_tokens,     # Number of tokens
    dim_model,      # Model dimension (embedding dimension)
    stride_weight,  # Stride for weight matrix
    stride_out,     # Stride for output matrix
    BLOCK_N: tl.constexpr,  # Block size for sequence
    BLOCK_DMODEL: tl.constexpr  # Block size for model dimension
):
    pid = tl.program_id(0)  # Program ID for parallel execution
    block_start = pid * BLOCK_N  # Start of the block in sequence
    offsets_n = block_start + tl.arange(0, BLOCK_N)  # Offsets for sequence
    offsets_d = tl.arange(0, BLOCK_DMODEL)  # Offsets for model dimension

    # Load token IDs for the current block
    token_ids = tl.load(token_ids_ptr + offsets_n, mask=offsets_n < num_tokens, other=0)

    # Calculate the starting position in the weight matrix for each token
    weight_offsets = token_ids[:, None] * stride_weight + offsets_d[None, :]

    # Load embeddings from the weight matrix
    embeddings = tl.load(weight_ptr + weight_offsets, mask=offsets_n[:, None] < num_tokens, other=0.0)

    # Calculate output offsets
    out_offsets = block_start * stride_out + offsets_d

    # Store embeddings in the output tensor
    tl.store(out_ptr + out_offsets, embeddings, mask=offsets_n[:, None] < num_tokens)


def embedding(token_ids: torch.Tensor, weight: torch.Tensor, out: torch.Tensor):
    num_tokens = token_ids.shape[0]
    dim_model = weight.shape[1]
    
    # Determine BLOCK_DMODEL as the next power of two
    BLOCK_DMODEL = 2 ** (dim_model - 1).bit_length()
    
    # Define the block size for sequence processing
    BLOCK_N = 128  # You can adjust this based on your needs and GPU capacity
    
    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(num_tokens, meta['BLOCK_N']),)
    embedding_kernel[grid](
        token_ids_ptr=token_ids,
        weight_ptr=weight,
        out_ptr=out,
        num_tokens=num_tokens,
        dim_model=dim_model,
        stride_weight=weight.stride(0),
        stride_out=out.stride(0),
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL
    )
