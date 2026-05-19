import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    token_ids_ptr,  # Pointer to the token IDs
    weight_ptr,     # Pointer to the weight matrix
    out_ptr,        # Pointer to the output tensor
    n_tokens,       # Number of tokens in the sequence
    n_sequences,    # Number of sequences
    d_model,        # Dimension of the embedding vectors
    stride_token,   # Stride for token IDs
    stride_weight,  # Stride for weight matrix
    stride_out,     # Stride for output tensor
    BLOCK_N: tl.constexpr,  # Block size for sequences
    BLOCK_DMODEL: tl.constexpr  # Block size for embedding dimensions
):
    # Get the program ID
    pid = tl.program_id(0)
    
    # Compute the starting index for this program
    start_n = pid * BLOCK_N
    end_n = start_n + BLOCK_N
    start_d = 0
    end_d = d_model
    
    # Compute the range of sequences to process
    token_ids = tl.load(token_ids_ptr + start_n, mask=start_n < n_tokens, other=0)
    
    # Iterate over the sequences
    for n in range(start_n, end_n):
        if n < n_tokens:
            token_id = token_ids[n % BLOCK_N]
            # Compute the offset in the weight matrix
            weight_offset = token_id * d_model
            # Load the embedding vector
            embedding = tl.load(weight_ptr + weight_offset + start_d, mask=start_d < d_model, other=0.0)
            # Store the embedding vector in the output tensor
            out_offset = n * d_model
            tl.store(out_ptr + out_offset + start_d, embedding, mask=start_d < d_model)

import torch
import triton
import triton.language as tl

def embedding(token_ids, weight, out):
    # Get the dimensions
    n_tokens, n_sequences = token_ids.shape
    d_model = weight.shape[1]
    
    # Define the block sizes
    BLOCK_N = 128
    BLOCK_DMODEL = 128
    
    # Align BLOCK_DMODEL to the next power of two
    BLOCK_DMODEL = 2 ** ((BLOCK_DMODEL - 1).bit_length())
    
    # Compute the grid size
    grid = (triton.cdiv(n_tokens, BLOCK_N),)
    
    # Define the strides
    stride_token = token_ids.stride(0)
    stride_weight = weight.stride(0)
    stride_out = out.stride(0)
    
    # Launch the kernel
    embedding_kernel[grid](
        token_ids, weight, out,
        n_tokens, n_sequences, d_model,
        stride_token, stride_weight, stride_out,
        BLOCK_N, BLOCK_DMODEL
    )

# Example usage
n_tokens = 1024
n_sequences = 64
d_model = 512

token_ids = torch.randint(0, 1000, (n_tokens, n_sequences), device='cuda')
weight = torch.randn(1000, d_model, device='cuda')
out = torch.empty(n_tokens, d_model, device='cuda')

embedding(token_ids, weight, out)
