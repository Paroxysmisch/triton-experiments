import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    # Pointers to tensors
    weight_ptr,    # Pointer to weight matrix [vocab_size, hidden_dim]
    input_ids_ptr, # Pointer to input token ids [batch_size, seq_len]
    output_ptr,    # Pointer to output tensor [batch_size, seq_len, hidden_dim]
    
    # Dimensions
    hidden_dim,    # Size of embedding dimension
    seq_len,       # Sequence length
    vocab_size,    # Size of vocabulary
    
    # Strides for memory access
    weight_stride,     # Stride for weight matrix
    input_ids_stride,  # Stride for input ids
    output_stride_b,   # Batch stride for output
    output_stride_s,   # Sequence stride for output
    
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,  # Block size for batch * seq dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for hidden dimension
):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate batch and sequence indices
    batch_seq_idx = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    batch_idx = batch_seq_idx // seq_len
    seq_idx = batch_seq_idx % seq_len
    
    # Calculate offset for hidden dimension
    hidden_offset = tl.arange(0, BLOCK_SIZE_N)
    
    # Create a mask for valid batch/sequence indices
    mask = batch_seq_idx < (seq_len * (pid + 1))
    
    # Load token IDs for current batch/sequence positions
    token_ids = tl.load(
        input_ids_ptr + batch_idx * input_ids_stride + seq_idx,
        mask=mask,
        other=-1
    )
    
    # Calculate weight matrix offsets based on token IDs
    weight_offset = token_ids[:, None] * weight_stride + hidden_offset[None, :]
    
    # Load embedding vectors for the tokens
    embeddings = tl.load(
        weight_ptr + weight_offset,
        mask=mask[:, None] & (token_ids[:, None] >= 0) & (token_ids[:, None] < vocab_size),
        other=0.0
    )
    
    # Calculate output offset
    output_offset = (
        batch_idx[:, None] * output_stride_b +
        seq_idx[:, None] * output_stride_s +
        hidden_offset[None, :]
    )
    
    # Store the embeddings in output tensor
    tl.store(
        output_ptr + output_offset,
        embeddings,
        mask=mask[:, None]
    )

def embedding(weight: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Efficient embedding lookup using Triton.
    
    Args:
        weight: Embedding weight matrix of shape [vocab_size, hidden_dim]
        input_ids: Input token IDs of shape [batch_size, seq_len]
    
    Returns:
        Output tensor of shape [batch_size, seq_len, hidden_dim]
    """
    batch_size, seq_len = input_ids.shape
    vocab_size, hidden_dim = weight.shape
    
    # Create output tensor
    output = torch.empty(
        (batch_size, seq_len, hidden_dim),
        device=weight.device,
        dtype=weight.dtype
    )
    
    # Calculate strides
    weight_stride = weight.stride(0)
    input_ids_stride = input_ids.stride(0)
    output_stride_b = output.stride(0)
    output_stride_s = output.stride(1)
    
    # Define block sizes
    BLOCK_SIZE_M = 32  # For batch * seq dimension
    BLOCK_SIZE_N = 64  # For hidden dimension
    
    # Calculate grid size
    grid = ((batch_size * seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,)
    
    # Launch kernel
    embedding_kernel[grid](
        weight_ptr=weight,
        input_ids_ptr=input_ids,
        output_ptr=output,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        vocab_size=vocab_size,
        weight_stride=weight_stride,
        input_ids_stride=input_ids_stride,
        output_stride_b=output_stride_b,
        output_stride_s=output_stride_s,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output
