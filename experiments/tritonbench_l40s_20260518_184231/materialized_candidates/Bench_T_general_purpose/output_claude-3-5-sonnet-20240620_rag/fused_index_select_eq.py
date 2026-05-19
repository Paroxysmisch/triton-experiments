import torch
import triton
import triton.language as tl

@triton.jit
def index_select_kernel(
    inp, out, M, N, index, index_len, other, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Get program ids for x and y axes
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    
    # Calculate row and column offsets
    rows_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offsets < M
    cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)
    cols_mask = cols_offsets < N

    # Compute masks for blocks and output
    block_mask = rows_mask & cols_mask
    out_mask = rows_mask & (cols_offsets < index_len)

    # Load indices and compute offsets
    indices = tl.load(index + cols_offsets, mask=(cols_offsets < index_len), other=0)
    inp_off = rows_offsets * N + indices[None, :]
    
    # Load selected input
    selected = tl.load(inp + inp_off, mask=block_mask, other=0.0)
    
    # Perform element-wise comparison with 'other'
    if isinstance(other, float):
        comparison_result = selected == other
    else:
        comparison_result = selected == tl.load(other + rows_offsets, mask=block_mask, other=0.0)

    # Store the boolean result in output
    tl.store(out + inp_off, comparison_result, mask=out_mask)

def fused_index_select_eq(input, dim, index, other, *, out=None):
    assert dim >= -input.ndim and dim < input.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"
    assert all((i >= 0 and i < input.size(dim)) for i in index), "Index out of range"

    # Adjust dimension and index
    if index.ndim == 0:
        index = index.unsqueeze(0)
    dim = dim % input.ndim
    input_shape = list(input.shape)
    index_len = index.numel()

    # Compress input along the dimension
    input = dim_compress(input, dim)
    N = input_shape[dim]
    M = input.numel() // N
    out_shape = list(input.shape)
    out_shape[input.ndim - 1] = index_len
    out = torch.empty(out_shape, dtype=torch.bool, device=input.device) if out is None else out

    # Define grid based on blocks
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(index_len, meta["BLOCK_N"]),
    )
    
    # Call the kernel with calculated grid
    index_select_kernel[grid](input, out, M, N, index, index_len, other)
    
    return out
