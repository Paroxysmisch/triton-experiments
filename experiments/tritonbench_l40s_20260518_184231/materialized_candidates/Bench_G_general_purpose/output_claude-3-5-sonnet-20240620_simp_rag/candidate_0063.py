import triton
import triton.language as tl
import torch

@triton.jit
def embedding_kernel(
    # Pointers to matrices
    weight_ptr,          # Pointer to weight matrix [vocab_size, embedding_dim]
    input_ids_ptr,       # Pointer to input IDs [batch_size, seq_len]
    output_ptr,         # Pointer to output tensor [batch_size, seq_len, embedding_dim]
    
    # Matrix dimensions
    seq_length,         # Sequence length
    vocab_size,         # Size of vocabulary
    embedding_dim,      # Dimension of embeddings
    
    # Strides for memory access
    weight_stride,      # Stride for weight matrix
    ids_stride,         # Stride for input IDs
    output_batch_stride,# Stride for output batch dimension
    output_seq_stride,  # Stride for output sequence dimension
    
    # Block sizes for parallelization
    BLOCK_SIZE: tl.constexpr,  # Size of blocks for parallel processing
):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate batch and sequence indices
    batch_idx = pid // seq_length
    seq_idx = pid % seq_length
    
    # Load input token ID
    token_id = tl.load(input_ids_ptr + batch_idx * ids_stride + seq_idx)
    
    # Check if token ID is valid
    is_valid = token_id < vocab_size
    
    # Calculate offsets
    weight_offset = token_id * weight_stride
    output_offset = (batch_idx * output_batch_stride + 
                    seq_idx * output_seq_stride)
    
    # Load and store embedding vectors in blocks
    for i in range(0, embedding_dim, BLOCK_SIZE):
        block_mask = i + tl.arange(0, BLOCK_SIZE) < embedding_dim
        mask = is_valid & block_mask
        
        # Load embedding vector block
        weight_block_ptr = weight_ptr + weight_offset + i
        vector = tl.load(weight_block_ptr, mask=mask)
        
        # Store to output
        output_block_ptr = output_ptr + output_offset + i
        tl.store(output_block_ptr, vector, mask=mask)

def embedding(weight: torch.Tensor, 
             input_ids: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the embedding kernel
    
    Args:
        weight: Embedding weight matrix [vocab_size, embedding_dim]
        input_ids: Input token IDs [batch_size, seq_len]
    
    Returns:
        output: Embedded vectors [batch_size, seq_len, embedding_dim]
    """
    batch_size, seq_length = input_ids.shape
    vocab_size, embedding_dim = weight.shape
    
    # Create output tensor
    output = torch.empty((batch_size, seq_length, embedding_dim),
                        device=weight.device,
                        dtype=weight.dtype)
    
    # Calculate strides
    weight_stride = weight.stride(0)
    ids_stride = input_ids.stride(0)
    output_batch_stride = output.stride(0)
    output_seq_stride = output.stride(1)
    
    # Configure kernel parameters
    BLOCK_SIZE = 32
    grid = (batch_size * seq_length,)
    
    # Launch kernel
    embedding_kernel[grid](
        weight_ptr=weight,
        input_ids_ptr=input_ids,
        output_ptr=output,
        seq_length=seq_length,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        weight_stride=weight_stride,
        ids_stride=ids_stride,
        output_batch_stride=output_batch_stride,
        output_seq_stride=output_seq_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
