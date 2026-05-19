import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    indices_ptr,
    weight_ptr,
    out_ptr,
    B,
    S,
    D,
    V,
    stride_indices_b,
    stride_indices_s,
    stride_weight_v,
    stride_weight_d,
    stride_out_b,
    stride_out_s,
    stride_out_d,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # 2D grid: token blocks vs embedding blocks
    pid_token = tl.program_id(0)
    pid_dmodel = tl.program_id(1)

    # Compute token offsets for this block
    token_start = pid_token * BLOCK_N
    token_offsets = token_start + tl.arange(0, BLOCK_N)
    
    # Calculate batch and sequence indices
    batch_idx = token_offsets // S
    seq_idx = token_offsets % S
    
    # Mask for valid tokens (within B*S)
    mask_tokens = (batch_idx < B) & (seq_idx < S)
    
    # Load token IDs from indices tensor
    indices_offset = batch_idx * stride_indices_b + seq_idx * stride_indices_s
    token_ids = tl.load(indices_ptr + indices_offset, mask=mask_tokens, other=0)
    
    # Additional mask for valid token IDs
    mask_tokens &= (token_ids >= 0) & (token_ids < V)

    # Compute embedding dimension offsets
    d_start = pid_dmodel * BLOCK_DMODEL
    d_offsets = d_start + tl.arange(0, BLOCK_DMODEL)
    mask_d = d_offsets < D

    # Broadcast dimensions for vectorized operations
    token_ids_matrix = tl.broadcast_to(token_ids[:, None], (BLOCK_N, BLOCK_DMODEL))
    d_offsets_matrix = tl.broadcast_to(d_offsets[None, :], (BLOCK_N, BLOCK_DMODEL))
    combined_mask = mask_tokens[:, None] & mask_d[None, :]

    # Calculate weight matrix offsets and load
    weight_offsets = token_ids_matrix * stride_weight_v + d_offsets_matrix * stride_weight_d
    embeddings = tl.load(weight_ptr + weight_offsets, mask=combined_mask, other=0.0)

    # Compute output tensor offsets
    batch_matrix = tl.broadcast_to(batch_idx[:, None], (BLOCK_N, BLOCK_DMODEL))
    seq_matrix = tl.broadcast_to(seq_idx[:, None], (BLOCK_N, BLOCK_DMODEL))
    out_offsets = (batch_matrix * stride_out_b + 
                   seq_matrix * stride_out_s + 
                   d_offsets_matrix * stride_out_d)

    # Store results with combined mask
    tl.store(out_ptr + out_offsets, embeddings, mask=combined_mask)

def embedding(indices, weight):
    # Validate input dimensions
    assert indices.dim() == 2, "Indices must be 2D (batch, sequence)"
    assert weight.dim() == 2, "Weight must be 2D (vocab, features)"
    
    B, S = indices.shape
    V, D = weight.shape
    
    # Preallocate output tensor
    out = torch.empty((B, S, D), device=indices.device, dtype=weight.dtype)
    
    # Configure block sizes
    BLOCK_N = 128  # Tune based on hardware characteristics
    BLOCK_DMODEL = triton.next_power_of_2(D)
    if BLOCK_DMODEL > 1024:  # Respect hardware limits
        BLOCK_DMODEL = 1024
    
    # Compute grid dimensions
    grid_token = (B * S + BLOCK_N - 1) // BLOCK_N
    grid_dmodel = (D + BLOCK_DMODEL - 1) // BLOCK_DMODEL
    grid = (grid_token, grid_dmodel)

    # Launch kernel with optimized parameters
    embedding_kernel[grid](
        indices, weight, out,
        B, S, D, V,
        indices.stride(0), indices.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
    )
    
    return out
