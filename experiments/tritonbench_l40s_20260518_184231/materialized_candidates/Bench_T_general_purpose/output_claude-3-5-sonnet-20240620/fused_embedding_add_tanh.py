import torch
import triton
import triton.language as tl
import math

@triton.jit
def fused_embedding_add_tanh_kernel(
    # Pointers to tensors
    output_ptr,      # Output tensor pointer
    indices_ptr,     # Input indices pointer
    weight_ptr,      # Embedding weight matrix pointer
    other_ptr,       # Other tensor to add pointer
    # Dimensions and metadata
    batch_size,      # Total number of indices
    vocab_size,      # Number of embeddings (V)
    embed_dim,       # Embedding dimension (D)
    padding_idx,     # Padding index value (-1 if not used)
    max_norm,        # Maximum norm value (0.0 if not used)
    stride_out,      # Output tensor stride
    stride_weight,   # Weight matrix stride
    stride_other,    # Other tensor stride
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute the index and embedding dimension offsets
    idx_offset = pid * BLOCK_SIZE
    
    # Load the index value
    idx = tl.load(indices_ptr + pid) if pid < batch_size else 0
    
    # Handle padding idx
    if padding_idx >= 0:
        mask = idx != padding_idx
    else:
        mask = 1
        
    # Compute row offset in weight matrix
    weight_row_offset = idx * stride_weight
    
    # Initialize accumulator for embedding vector
    offs_em = tl.arange(0, BLOCK_SIZE)
    mask_em = offs_em < embed_dim
    
    # Load embedding vector
    embed = tl.load(weight_ptr + weight_row_offset + offs_em, mask=mask_em)
    
    # Apply max norm if specified
    if max_norm > 0.0:
        norm = tl.sqrt(tl.sum(embed * embed))
        scale = tl.where(norm > max_norm, max_norm / norm, 1.0)
        embed = embed * scale
    
    # Load other tensor
    other = tl.load(other_ptr + offs_em, mask=mask_em)
    
    # Compute fused add and tanh
    output = tl.tanh(embed + other)
    
    # Write result
    output_row_offset = pid * stride_out
    tl.store(output_ptr + output_row_offset + offs_em, 
             output * mask, mask=mask_em)

def fused_embedding_add_tanh(
    input_indices: torch.Tensor,
    weight: torch.Tensor,
    other: torch.Tensor,
    *,
    padding_idx: int = None,
    max_norm: float = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
    out: torch.Tensor = None
) -> torch.Tensor:
    """
    Fused operation combining embedding lookup, element-wise addition, and tanh activation.
    
    Args:
        input_indices (torch.Tensor): Indices tensor of shape (*)
        weight (torch.Tensor): Embedding matrix of shape (V, D)
        other (torch.Tensor): Tensor to add, must be broadcastable
        padding_idx (int, optional): Index for padding
        max_norm (float, optional): Max norm for embeddings
        norm_type (float, optional): Type of norm (only 2.0 supported in Triton)
        scale_grad_by_freq (bool, optional): Whether to scale grads by frequency
        sparse (bool, optional): Whether to use sparse gradients
        out (torch.Tensor, optional): Output tensor
        
    Returns:
        torch.Tensor: Result of shape (*) + (D,)
    """
    # Input validation
    assert norm_type == 2.0, "Only L2 norm is supported in Triton implementation"
    assert weight.dim() == 2, "Weight matrix must be 2-dimensional"
    
    # Get dimensions
    vocab_size, embed_dim = weight.shape
    batch_size = input_indices.numel()
    
    # Prepare output tensor
    if out is None:
        output_shape = input_indices.shape + (embed_dim,)
        out = torch.empty(output_shape, device=input_indices.device, 
                         dtype=weight.dtype)
    
    # Prepare other tensor for broadcasting
    other = other.expand_as(out)
    
    # Convert padding_idx to integer
    padding_idx_int = -1 if padding_idx is None else padding_idx
    max_norm_float = 0.0 if max_norm is None else float(max_norm)
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(batch_size, meta['BLOCK_SIZE']),)
    
    fused_embedding_add_tanh_kernel[grid](
        out.data_ptr(),
        input_indices.data_ptr(),
        weight.data_ptr(),
        other.data_ptr(),
        batch_size,
        vocab_size,
        embed_dim,
        padding_idx_int,
        max_norm_float,
        out.stride(-1),
        weight.stride(0),
        other.stride(-1),
        BLOCK_SIZE=128,
    )
    
    return out
