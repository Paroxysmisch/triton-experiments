import triton
import triton.language as tl
import torch

@triton.jit
def embedding_kernel(
    weight_ptr,        # Pointer to the embedding weight matrix
    input_ids_ptr,     # Pointer to the input token IDs
    out_ptr,           # Pointer to the output tensor
    num_embeddings,    # Total number of embeddings
    embedding_dim,     # Dimension of each embedding vector
    batch_size,        # Batch size
    seq_len,           # Sequence length
    stride_w,          # Stride for the weight matrix
    stride_o,          # Stride for the output matrix
    BLOCK_SIZE: tl.constexpr,  # Block size for the kernel
):
    # Program ID for parallelization
    pid = tl.program_id(axis=0)

    # Compute the start index for the batch and sequence
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len

    # Load the input token ID
    input_id = tl.load(input_ids_ptr + batch_idx * seq_len + seq_idx)

    # Check if the input_id is valid
    if input_id < num_embeddings:
        # Compute the start pointer for the embedding vector
        weight_offset = input_id * stride_w
        out_offset = (batch_idx * seq_len + seq_idx) * stride_o

        # Load the embedding vector
        for i in range(0, embedding_dim, BLOCK_SIZE):
            # Load a block of the embedding vector
            weight_block = tl.load(weight_ptr + weight_offset + i + tl.arange(0, BLOCK_SIZE))
            # Store the block in the output tensor
            tl.store(out_ptr + out_offset + i + tl.arange(0, BLOCK_SIZE), weight_block)

def embedding(weight, input_ids, out):
    # Get tensor dimensions
    batch_size, seq_len = input_ids.shape
    num_embeddings, embedding_dim = weight.shape

    # Define grid size
    grid = (batch_size * seq_len,)

    # Launch the Triton kernel
    embedding_kernel[grid](
        weight_ptr=weight,
        input_ids_ptr=input_ids,
        out_ptr=out,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        seq_len=seq_len,
        stride_w=weight.stride(0),
        stride_o=out.stride(0),
        BLOCK_SIZE=32,  # Assuming a block size of 32
    )

# Example usage
weight = torch.randn(10000, 768, device='cuda')  # Example weight matrix
input_ids = torch.randint(0, 10000, (32, 128), device='cuda')  # Example input IDs
out = torch.empty((32, 128, 768), device='cuda')  # Output tensor

embedding(weight, input_ids, out)
