import torch
import triton
import triton.language as tl

@triton.jit
def fused_index_select_eq_kernel(
    inp, out, other, M, N, index, index_len, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Get program ids for x and y axes
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    
    # Calculate row and column offsets
    rows_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offsets < M
    cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)
    cols_mask = cols_offsets < index_len

    # Compute masks for blocks and output
    block_mask = rows_mask & cols_mask

    # Load indices and compute offsets
    indices = tl.load(index + cols_offsets, mask=cols_mask, other=0)
    inp_off = rows_offsets * N + indices[None, :]
    out_off = rows_offsets * index_len + cols_offsets[None, :]

    # Load selected input
    selected = tl.load(inp + inp_off, mask=block_mask, other=0.0)

    # Perform element-wise equality comparison
    comparison = selected == tl.load(other, mask=block_mask, other=0.0)

    # Store the result in the output
    tl.store(out + out_off, comparison, mask=block_mask)


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

    if out is None:
        out = torch.empty(out_shape, dtype=torch.bool, device=input.device)

    # Broadcast other to match the shape of the selected elements
    if isinstance(other, (int, float)):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    other = other.expand(out_shape)

    # Define grid based on blocks
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(index_len, meta["BLOCK_N"]),
    )
    
    # Call the kernel with calculated grid
    fused_index_select_eq_kernel[grid](input, out, other, M, N, index, index_len)
    
    # Adjust output order if necessary
    if dim != out.ndim - 1:
        order = [i for i in range(out.ndim - 1)]
        order.insert(dim, out.ndim - 1)
        return out.permute(order)
    else:
        return out

# Helper function to compress input along a given dimension
def dim_compress(tensor, dim):
    shape = tensor.shape
    return tensor.permute([dim] + [i for i in range(len(shape)) if i != dim]).reshape(shape[dim], -1)
