import triton
import triton.language as tl
import torch

@triton.jit
def embedding_kernel(
    weight_ptr,
    input_ids_ptr,
    out_ptr,
    weight_stride,
    out_batch_stride,
    out_seq_stride,
    out_emb_stride,
    num_embeddings: tl.constexpr,
    embedding_dim: tl.constexpr,
    batch_size: tl.constexpr,
    seq_length: tl.constexpr,
    BLOCK_SIZE_SEQ: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_SEQ

    # Compute the range of sequence indices for this block
    seq_start = block_start
    seq_end = min(seq_start + BLOCK_SIZE_SEQ, seq_length)

    # Iterate over the sequence indices
    for seq_idx in range(seq_start, seq_end):
        # Load the input ID for this sequence index
        input_id = tl.load(input_ids_ptr + seq_idx)

        # Compute the base pointer for the embedding vector
        weight_base_ptr = weight_ptr + input_id * embedding_dim

        # Iterate over the batch dimension
        for batch_idx in range(batch_size):
            # Compute the output pointer for this batch and sequence index
            out_base_ptr = out_ptr + batch_idx * out_batch_stride + seq_idx * out_seq_stride

            # Load the embedding vector and store it in the output tensor
            for emb_idx in range(embedding_dim):
                out_ptr_final = out_base_ptr + emb_idx * out_emb_stride
                weight_ptr_final = weight_base_ptr + emb_idx
                tl.store(out_ptr_final, tl.load(weight_ptr_final))

def embedding(
    weight: torch.Tensor,
    input_ids: torch.Tensor,
    out: torch.Tensor,
    max_grid: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
):
    assert weight.is_cuda and input_ids.is_cuda and out.is_cuda, "Tensors must be on CUDA device"
    assert weight.dim() == 2, "Weight tensor must be 2D"
    assert input_ids.dim() == 1, "Input IDs tensor must be 1D"
    assert out.dim() == 3, "Output tensor must be 3D"
    assert out.shape[1] == input_ids.shape[0], "Output sequence length must match input IDs length"
    assert out.shape[2] == weight.shape[1], "Output embedding dimension must match weight embedding dimension"

    num_embeddings = weight.shape[0]
    embedding_dim = weight.shape[1]
    batch_size = out.shape[0]
    seq_length = out.shape[1]

    weight_stride = weight.stride(0)
    out_batch_stride = out.stride(0)
    out_seq_stride = out.stride(1)
    out_emb_stride = out.stride(2)

    # Define the grid and block dimensions
    BLOCK_SIZE_SEQ = 128
    grid = (triton.cdiv(seq_length, BLOCK_SIZE_SEQ),)

    # Launch the kernel
    embedding_kernel[grid](
        weight_ptr=weight,
        input_ids_ptr=input_ids,
        out_ptr=out,
        weight_stride=weight_stride,
        out_batch_stride=out_batch_stride,
        out_seq_stride=out_seq_stride,
        out_emb_stride=out_emb_stride,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        seq_length=seq_length,
        BLOCK_SIZE_SEQ=BLOCK_SIZE_SEQ,
    )
