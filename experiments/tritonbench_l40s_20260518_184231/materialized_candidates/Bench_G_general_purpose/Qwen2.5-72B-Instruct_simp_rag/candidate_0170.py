import torch
import triton
import triton.language as tl

# Triton kernel for index select
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
    cols_mask = cols_offsets < index_len

    # Compute masks for blocks and output
    block_mask = rows_mask & cols_mask
    out_mask = rows_mask & cols_mask

    # Load indices and compute offsets
    indices = tl.load(index + cols_offsets, mask=cols_mask, other=0)
    inp_off = rows_offsets * N + indices[None, :]
    out_off = rows_offsets * index_len + cols_offsets[None, :]

    # Load selected input and store in output
    selected = tl.load(inp + inp_off, mask=block_mask, other=0.0)
    tl.store(out + out_off, selected, mask=out_mask)

# Wrapper function for index select
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

# Helper function to compress input along a dimension
def dim_compress(tensor, dim):
    shape = list(tensor.shape)
    shape.pop(dim)
    shape.append(tensor.shape[dim])
    return tensor.view(shape)

# Example usage
if __name__ == "__main__":
    # Create input tensor
    inp = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device="cuda")
    # Create index tensor
    index = torch.tensor([0, 2], device="cuda")
    # Select rows based on index
    out = index_select(inp, 0, index)
    print(out)
