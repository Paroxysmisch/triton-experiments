import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    weight_ptr,  # Pointer to the embedding weight matrix
    input_ids_ptr,  # Pointer to the input token IDs
    out_ptr,  # Pointer to the output tensor
    weight_stride,  # Stride of the weight matrix
    input_ids_stride,  # Stride of the input IDs
    out_stride,  # Stride of the output tensor
    num_tokens,  # Number of tokens in the input
    embedding_dim,  # Dimension of the embedding vectors
    BLOCK_SIZE: tl.constexpr  # Block size for parallelism
):
    # Compute the block ID
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Iterate over the input tokens
    for i in range(block_start, min(block_start + BLOCK_SIZE, num_tokens)):
        # Load the token ID
        token_id = tl.load(input_ids_ptr + i * input_ids_stride)

        # Compute the output index
        out_index = i * out_stride

        # Load the embedding vector
        for j in range(embedding_dim):
            weight_index = token_id * weight_stride + j
            out_index_j = out_index + j
            tl.store(out_ptr + out_index_j, tl.load(weight_ptr + weight_index))

import torch
import triton
import triton.language as tl

def embedding(weight, input_ids, out):
    # Extract dimensions
    num_tokens = input_ids.shape[0]
    embedding_dim = weight.shape[1]

    # Define grid and block dimensions
    BLOCK_SIZE = 128
    grid = (triton.cdiv(num_tokens, BLOCK_SIZE),)

    # Define strides
    weight_stride = weight.stride(1)
    input_ids_stride = input_ids.stride(0)
    out_stride = out.stride(1)

    # Launch the kernel
    embedding_kernel[grid](
        weight.data_ptr(),  # Pointer to the embedding weight matrix
        input_ids.data_ptr(),  # Pointer to the input token IDs
        out.data_ptr(),  # Pointer to the output tensor
        weight_stride,  # Stride of the weight matrix
        input_ids_stride,  # Stride of the input IDs
        out_stride,  # Stride of the output tensor
        num_tokens,  # Number of tokens in the input
        embedding_dim,  # Dimension of the embedding vectors
        BLOCK_SIZE  # Block size for parallelism
    )

# Example usage
if __name__ == "__main__":
    # Example input
    num_tokens = 1024
    embedding_dim = 512
    weight = torch.randn((10000, embedding_dim), device='cuda')
    input_ids = torch.randint(0, 10000, (num_tokens,), device='cuda')
    out = torch.empty((num_tokens, embedding_dim), device='cuda')

    # Compute embeddings
    embedding(weight, input_ids, out)

    # Print the first few embeddings
    print(out[:5])
