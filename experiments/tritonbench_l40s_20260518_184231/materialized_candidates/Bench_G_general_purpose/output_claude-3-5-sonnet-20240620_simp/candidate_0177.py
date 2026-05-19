import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'NUM_WARPS': 4}),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 8}),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 16}),
    ],
    key=['n_cols']
)
@triton.jit
def chunk_global_cumsum_vector_kernel(
    output_ptr, input_ptr,
    n_rows, n_cols, chunk_size,
    stride_out_row, stride_out_col,
    stride_in_row, stride_in_col,
    BLOCK_SIZE: tl.constexpr
):
    # Compute indices for this program instance
    pid = tl.program_id(0)
    row_idx = pid // (n_cols // BLOCK_SIZE)
    col_start = (pid % (n_cols // BLOCK_SIZE)) * BLOCK_SIZE

    # Create block mask for loading/storing
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < (n_cols - col_start)
    
    # Load input block
    row_ptr = input_ptr + row_idx * stride_in_row + col_start * stride_in_col
    x = tl.load(row_ptr + col_offsets * stride_in_col, mask=mask)
    
    # Compute cumsum within chunk
    chunk_idx = col_offsets // chunk_size
    chunk_offset = col_offsets % chunk_size
    
    # Create triangular mask for cumsum
    tri_mask = chunk_offset[:, None] >= chunk_offset[None, :]
    tri_mask = tri_mask & mask[:, None] & mask[None, :]
    
    # Perform cumsum using matrix multiplication
    chunk_result = tl.dot(tri_mask, x, x)
    
    # Store result
    out_ptr = output_ptr + row_idx * stride_out_row + col_start * stride_out_col
    tl.store(out_ptr + col_offsets * stride_out_col, chunk_result, mask=mask)

def chunk_global_cumsum_vector(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Compute chunk-based cumulative sum across a vector.
    
    Args:
        x: Input tensor of shape (batch_size, n_cols)
        chunk_size: Size of chunks for cumulative sum
        
    Returns:
        Tensor of same shape as input with chunk-wise cumulative sums
    """
    assert x.dim() == 2, "Input tensor must be 2-dimensional"
    batch_size, n_cols = x.shape
    
    # Ensure chunk size divides n_cols evenly
    assert n_cols % chunk_size == 0, "n_cols must be divisible by chunk_size"
    
    # Prepare output tensor
    output = torch.empty_like(x)
    
    # Launch kernel
    grid = (triton.cdiv(n_cols, 128) * batch_size,)
    chunk_global_cumsum_vector_kernel[grid](
        output, x,
        batch_size, n_cols, chunk_size,
        output.stride(0), output.stride(1),
        x.stride(0), x.stride(1),
        BLOCK_SIZE=128
    )
    
    return output
