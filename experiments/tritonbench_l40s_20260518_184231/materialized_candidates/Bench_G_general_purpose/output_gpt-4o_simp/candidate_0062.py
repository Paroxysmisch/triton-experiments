import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    weight_ptr, input_ids_ptr, out_ptr,
    num_embeddings, embedding_dim,
    stride_w, stride_o, num_ids,
    BLOCK_SIZE: tl.constexpr
):
    # Block index
    pid = tl.program_id(0)

    # Calculate the start position for this block
    start = pid * BLOCK_SIZE

    # Loop over each token ID in the block
    for i in range(BLOCK_SIZE):
        # Calculate the index for this particular thread
        idx = start + i

        # Check if this index is within the range of input IDs
        if idx < num_ids:
            # Load the token ID
            token_id = tl.load(input_ids_ptr + idx)

            # Check if the token ID is valid
            if token_id < num_embeddings:
                # Calculate the position in the weight matrix
                offset_w = token_id * stride_w

                # Calculate the position in the output matrix
                offset_o = idx * stride_o

                # Load the embedding vector and store it in the output
                for j in range(embedding_dim):
                    embedding_val = tl.load(weight_ptr + offset_w + j)
                    tl.store(out_ptr + offset_o + j, embedding_val)

def embedding(weight, input_ids, out, BLOCK_SIZE=128):
    # Get dimensions and strides
    num_ids = input_ids.shape[0]
    num_embeddings, embedding_dim = weight.shape
    stride_w = weight.stride(0)
    stride_o = out.stride(0)

    # Define grid size
    grid = (triton.cdiv(num_ids, BLOCK_SIZE),)

    # Launch the kernel
    embedding_kernel[grid](
        weight, input_ids, out,
        num_embeddings, embedding_dim,
        stride_w, stride_o, num_ids,
        BLOCK_SIZE=BLOCK_SIZE
    )

# Example usage
# weight = torch.randn(num_embeddings, embedding_dim, device='cuda')
# input_ids = torch.randint(0, num_embeddings, (num_ids,), device='cuda')
# out = torch.empty(num_ids, embedding_dim, device='cuda')
# embedding(weight, input_ids, out)
