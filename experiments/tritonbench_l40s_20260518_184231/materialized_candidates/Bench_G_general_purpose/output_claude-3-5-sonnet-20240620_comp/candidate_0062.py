I'll help you create an efficient embedding kernel and wrapper using Triton. This implementation will handle embedding lookups efficiently on the GPU.

<antThinking>This code represents a substantial, reusable GPU kernel implementation that users may want to modify and tune. It's complex enough to warrant being an artifact and will likely be referenced or modified.</antThinking>

<Antartifact
identifier="embedding-triton-kernel"
type="application/vnd.ant.code"
language="python"
title="Triton Embedding Kernel Implementation">
import triton
import triton.language as tl
import torch

@triton.jit
def embedding_kernel(
    # Pointers to tensors
    output_ptr,    # Pointer to output tensor [B, S, D]
    weight_ptr,    # Pointer to weight matrix [V, D]
    indices_ptr,   # Pointer to input indices [B, S]
    # Dimensions
    batch_size,    # Batch size
    seq_len,       # Sequence length
    num_embed,     # Embedding dimension
    vocab_size,    # Vocabulary size
    # Strides for tensors
    output_batch_stride,
    output_seq_stride,
    weight_stride,
    indices_batch_stride,
    BLOCK_N: tl.constexpr,        # Block size for sequence dimension
    BLOCK_DMODEL: tl.constexpr,   # Block size for embedding dimension
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(seq_len, BLOCK_N)
    num_pid_d = tl.cdiv(num_embed, BLOCK_DMODEL)
    
    # Compute batch and sequence block indices
    pid_n = pid // num_pid_d
    pid_d = pid % num_pid_d
    
    # Compute batch and sequence indices
    batch_idx = pid_n // num_pid_n
    seq_idx = (pid_n % num_pid_n) * BLOCK_N
    
    # Compute embedding dimension offset
    dmodel_idx = pid_d * BLOCK_DMODEL
    
    # Create offset arrays
    offs_n = seq_idx + tl.arange(0, BLOCK_N)
    offs_d = dmodel_idx + tl.arange(0, BLOCK_DMODEL)
    
    # Create masks
    mask_n = offs_n < seq_len
    mask_d = offs_d < num_embed
    
    # Load token IDs for current sequence block
    token_ids = tl.load(
        indices_ptr + batch_idx * indices_batch_stride + offs_n,
        mask=mask_n,
        other=-1
    )
    
    # Compute output pointer offset
    output_off = (
        batch_idx * output_batch_stride +
        offs_n[:, None] * output_seq_stride +
        offs_d[None, :]
    )
    
    # Load and store embeddings
    mask = mask_n[:, None] & mask_d[None, :]
    for idx in range(BLOCK_N):
        if idx < seq_len:
            token_id = token_ids[idx]
            if token_id >= 0 and token_id < vocab_size:
                weight_off = token_id * weight_stride + offs_d
                embed = tl.load(weight_ptr + weight_off, mask=mask_d)
                tl.store(
                    output_ptr + output_off[idx],
                    embed,
                    mask=mask[idx]
                )

def embedding(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Efficient embedding lookup using Triton.
    
    Args:
        weight: Embedding weight matrix of shape [vocab_size, embed_dim]
        indices: Token indices of shape [batch_size, seq_len]
    
    Returns:
        Tensor of shape [batch_size, seq_len, embed_dim]
    """
    batch_size, seq_len = indices.shape
    vocab_size, embed_dim = weight.shape
    
    # Create output tensor
    output = torch.empty(
        (batch_size, seq_len, embed_dim),
        device=weight.device,
        dtype=weight.dtype
    )
    
    # Calculate optimal block sizes
    BLOCK_N = min(128, seq_len)
    BLOCK_DMODEL = min(256, triton.next_power_of_2(embed_dim))
    
    # Calculate grid size
    grid = (
        triton.cdiv(seq_len, BLOCK_N) * batch_size *
        triton.cdiv(embed_dim, BLOCK_DMODEL),
    )
    
    # Launch kernel
    embedding_kernel[grid](
        output,
        weight,
        indices,
        batch_size,
        seq_len,
        embed_dim,
        vocab_size,
        output.stride(0),
        output.stride(1),
        weight.stride(0),
        indices.stride(0),
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
    )
    
    return output
</Antartifact>

Would you like me to explain how this implementation works in detail? I can break down the key components and optimization strategies used in both the kernel and wrapper function.

The kernel is optimized for:
- Efficient memory access patterns
- Parallel processing across batch and sequence dimensions
- Configurable block sizes for tuning
- Proper masking for boundary conditions
- Minimal warp divergence
