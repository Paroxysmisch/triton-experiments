import torch
import triton
import triton.language as tl

@triton.jit
def index_select_kernel(
    inp, out, M, N, index, index_len, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
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
    out_off = rows_offsets * index_len + cols_offsets[None, :]

    # Load selected input and store in output
    selected = tl.load(inp + inp_off, mask=block_mask, other=0.0)
    tl.store(out + out_off, selected, mask=out_mask)


def index_select(inp, dim, index):
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"
    assert all((i >= 0 and i < inp.size(dim)) for i in index), "Index out of range"

    # Adjust dimension and index
    if index.ndim == 0:
        index = index.unsqueeze(0)
    dim = dim % inp.ndim
    inp_shape = list(inp.shape)
    index_len = index.numel()

    # Compress input along the dimension
    inp = dim_compress(inp, dim)
    N = inp_shape[dim]
    M = inp.numel() // N
    out_shape = list(inp.shape)
    out_shape[inp.ndim - 1] = index_len
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # Define grid based on blocks
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(index_len, meta["BLOCK_N"]),
    )
    
    # Call the kernel with calculated grid
    index_select_kernel[grid](inp, out, M, N, index, index_len)
    
    # Adjust output order if necessary
    if dim != out.ndim - 1:
        order = [i for i in range(out.ndim - 1)]
        order.insert(dim, out.ndim - 1)
        return out.permute(order)
    else:
        return out

@torch.inference_mode()
def fused_index_select_eq(input, dim, index, other, *, out=None):
    # Perform index selection
    selected = index_select(input, dim, index)
    
    # Broadcast other if it is a scalar
    if torch.is_tensor(other):
        assert selected.shape == other.shape, "Shapes must be broadcastable"
        pass
    else:
        other = torch.full(selected.shape, other, dtype=selected.dtype, device=selected.device)
    
    # Perform element-wise equality comparison
    result = selected == other
    
    # Return result as boolean tensor
    return result.to(torch.bool)
