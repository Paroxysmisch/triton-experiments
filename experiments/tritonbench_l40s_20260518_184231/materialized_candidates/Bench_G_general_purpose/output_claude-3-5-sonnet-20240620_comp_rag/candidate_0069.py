import triton
import triton.language as tl
import torch

@triton.jit
def embedding_kernel(
    weight_ptr, out_ptr, indices_ptr,
    seq_len, num_embed, embed_dim,
    weight_stride_0, weight_stride_1,
    out_stride_0, out_stride_1,
    BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute offsets
    num_blocks_n = triton.cdiv(seq_len, BLOCK_N)
    block_id_n = pid // num_blocks_n
    block_id_m = pid % num_blocks_n
    
    # Compute start offset for this block
    start_n = block_id_n * BLOCK_N
    start_m = block_id_m * BLOCK_DMODEL
    
    # Iterate over the sequence
    for idx in range(start_n, start_n + BLOCK_N):
        # Compute token ID and mask
        token_id = tl.load(indices_ptr + idx, mask=idx < seq_len, other=0)
        mask = idx < seq_len
        
        # Iterate over embedding dimension
        for dim in range(start_m, start_m + BLOCK_DMODEL, BLOCK_DMODEL):
            # Load weight
            weight = tl.load(
                weight_ptr + token_id * weight_stride_0 + dim * weight_stride_1,
                mask=mask & (dim < embed_dim),
                other=0.0
            )
            
            # Store output
            tl.store(
                out_ptr + idx * out_stride_0 + dim * out_stride_1,
                weight,
                mask=mask & (dim < embed_dim)
            )

def embedding(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    seq_len, num_embed = indices.shape
    _, embed_dim = weight.shape
    
    # Align BLOCK_DMODEL to the next power of 2
    BLOCK_DMODEL = triton.next_power_of_2(embed_dim)
    
    # Allocate output
    output = torch.empty((seq_len, num_embed, embed_dim), device=weight.device, dtype=weight.dtype)
    
    # Configure grid
    grid = lambda META: (
        triton.cdiv(seq_len, META['BLOCK_N']) * triton.cdiv(embed_dim, META['BLOCK_DMODEL']),
    )
    
    # Launch kernel
    embedding_kernel[grid](
        weight, output, indices,
        seq_len, num_embed, embed_dim,
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_N=32,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=2,
    )
    
    return output

# Example usage
if __name__ == "__main__":
    # Create sample data
    weight = torch.randn(1000, 128, device='cuda')
    indices = torch.randint(0, 1000, (64, 32), device='cuda')
    
    # Run embedding
    result = embedding(weight, indices)
    print(f"Output shape: {result.shape}")
