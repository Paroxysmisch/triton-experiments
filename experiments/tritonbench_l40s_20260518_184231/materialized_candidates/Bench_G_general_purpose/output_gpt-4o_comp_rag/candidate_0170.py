import torch
import triton
import triton.language as tl

@triton.jit
def index_select_cat_fwd_kernel(
    source, output, M, N, index, index_len, BLOCK_SIZE_INDEX: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr
):
    # Get program ids for x and y axes
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    
    # Calculate row and column offsets
    row_offsets = pid_x * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)[:, None]
    row_mask = row_offsets < M
    col_offsets = pid_y * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    col_mask = col_offsets < N

    # Compute masks for blocks and output
    block_mask = row_mask & col_mask
    out_mask = row_mask & (col_offsets < index_len)

    # Load indices and compute offsets
    indices = tl.load(index + col_offsets, mask=(col_offsets < index_len), other=0)
    source_offsets = row_offsets * N + indices[None, :]
    output_offsets = row_offsets * index_len + col_offsets[None, :]

    # Load selected source data and store in output
    selected = tl.load(source + source_offsets, mask=block_mask, other=0.0)
    tl.store(output + output_offsets, selected, mask=out_mask)


def index_select_cat_fwd(source, index):
    assert source.ndim == 2, "Source tensor must be 2-dimensional"
    assert index.ndim == 1, "Index tensor must be 1-dimensional"
    
    M, N = source.shape
    index_len = index.numel()

    # Check if the number of indices exceeds the number of rows in the source
    if index_len > M:
        print("Warning: Number of indices exceeds number of rows in the source. Truncating indices.")
        index = index[:M]
        index_len = M

    # Prepare the output tensor
    output_shape = (M, index_len)
    output = torch.empty(output_shape, dtype=source.dtype, device=source.device)

    # Define grid based on blocks
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_INDEX"]),
        triton.cdiv(index_len, meta["BLOCK_SIZE_COL"]),
    )

    # Call the kernel with calculated grid
    index_select_cat_fwd_kernel[grid](
        source, output, M, N, index, index_len,
        BLOCK_SIZE_INDEX=32,  # Example block size, can be tuned
        BLOCK_SIZE_COL=32     # Example block size, can be tuned
    )
    
    return output

# Example usage
source = torch.randn(128, 64, device='cuda')
index = torch.randint(0, 64, (32,), device='cuda')
output = index_select_cat_fwd(source, index)
print(output)
