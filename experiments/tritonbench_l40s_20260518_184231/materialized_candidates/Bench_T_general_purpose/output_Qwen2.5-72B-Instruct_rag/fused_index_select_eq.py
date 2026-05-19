import torch
import triton
import triton.language as tl

@triton.jit
def fused_index_select_eq_kernel(
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
    out_off = rows_offsets * index_len + cols_offsets[None, :]

    # Load selected input and other
    selected = tl.load(inp + inp_off, mask=block_mask, other=0.0)
    other_val = tl.load(other, mask=cols_mask, other=0.0)

    # Perform element-wise equality comparison
    result = selected == other_val

    # Store the result in the output tensor
    tl.store(out + out_off, result, mask=out_mask)

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
    input = input.permute(*[i for i in range(dim)] + [dim] + [i for i in range(dim + 1, input.ndim)])
    N = input_shape[dim]
    M = input.numel() // N
    out_shape = list(input.shape)
    out_shape[dim] = index_len

    # Create output tensor if not provided
    if out is None:
        out = torch.empty(out_shape, dtype=torch.bool, device=input.device)
    else:
        assert out.shape == out_shape, "Output tensor shape mismatch"
        assert out.dtype == torch.bool, "Output tensor must be of boolean type"

    # Define grid based on blocks
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(index_len, meta["BLOCK_N"]),
    )
    
    # Call the kernel with calculated grid
    fused_index_select_eq_kernel[grid](input, out, M, N, index, index_len, other)

    # Adjust output order if necessary
    if dim != input.ndim - 1:
        order = [i for i in range(input.ndim - 1)]
        order.insert(dim, input.ndim - 1)
        out = out.permute(order)

    return out
