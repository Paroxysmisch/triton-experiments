import torch
import triton
import triton.language as tl

@triton.jit
def fused_index_select_eq_kernel(
    inp_ptr, other_ptr, out_ptr, M, N, index_ptr, index_len,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    # Calculate row and column offsets for the current block
    row_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = row_offsets < M
    col_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = col_offsets < index_len

    # Combined mask for valid elements
    block_mask = row_mask & col_mask

    # Load indices for the current columns
    indices = tl.load(index_ptr + col_offsets, mask=col_mask, other=0)

    # Compute input pointers: inp_off = row * N + indices
    inp_offsets = row_offsets * N + indices[None, :]
    # Compute other and output pointers: row * index_len + col
    other_offsets = row_offsets * index_len + col_offsets[None, :]

    # Load input and other values
    selected = tl.load(inp_ptr + inp_offsets, mask=block_mask, other=0.0)
    other_val = tl.load(other_ptr + other_offsets, mask=block_mask, other=0.0)

    # Compute equality and store result
    result = selected == other_val
    tl.store(out_ptr + other_offsets, result, mask=block_mask)

def dim_compress(tensor, dim):
    dim = dim % tensor.ndim
    perm = list(range(tensor.ndim))
    perm.remove(dim)
    perm.append(dim)
    compressed = tensor.permute(perm)
    compressed = compressed.reshape(-1, tensor.size(dim))
    return compressed

def dim_decompress(compressed, dim, original_shape):
    original_dims = list(original_shape)
    idx_len = compressed.size(1)
    original_dims[dim] = idx_len
    decompressed_shape = original_dims[:dim] + original_dims[dim+1:] + [idx_len]
    decompressed = compressed.view(decompressed_shape)
    perm = list(range(decompressed.ndim))
    perm.insert(dim, perm.pop())
    decompressed = decompressed.permute(perm)
    return decompressed.contiguous()

def fused_index_select_eq(input, dim, index, other, *, out=None):
    # Validate inputs
    assert dim >= -input.ndim and dim < input.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have 1 or 0 dimensions"
    index = index.to(device=input.device)
    if index.numel() == 0:
        raise ValueError("Index tensor must not be empty")

    # Adjust index to be 1D
    if index.ndim == 0:
        index = index.unsqueeze(0)
    index_len = index.numel()
    dim = dim % input.ndim

    # Compute the shape of the selected tensor S
    s_shape = list(input.shape)
    s_shape[dim] = index_len
    s_shape = tuple(s_shape)

    # Broadcast 'other' to s_shape
    if isinstance(other, torch.Tensor):
        other = other.to(device=input.device)
        try:
            other_broadcasted = torch.broadcast_to(other, s_shape)
        except RuntimeError:
            raise RuntimeError(f"other tensor shape {other.shape} cannot be broadcasted to {s_shape}")
    else:
        other_broadcasted = torch.full(s_shape, other, dtype=input.dtype, device=input.device)

    # Compress input and other tensors along 'dim'
    input_compressed = dim_compress(input, dim)
    other_compressed = dim_compress(other_broadcasted, dim)
    M, N = input_compressed.shape
    assert other_compressed.shape == (M, index_len), "Compressed other shape mismatch"

    # Create output tensor
    out_compressed = torch.empty((M, index_len), dtype=torch.bool, device=input.device)

    # Define grid and block dimensions
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(index_len, BLOCK_N))

    # Launch kernel
    fused_index_select_eq_kernel[grid](
        input_compressed, other_compressed, out_compressed, M, N, index, index_len,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )

    # Decompress the output tensor
    out_decompressed = dim_decompress(out_compressed, dim, input.shape)

    # Handle output tensor
    if out is not None:
        if not out.dtype == torch.bool:
            raise TypeError("Output tensor must be of dtype torch.bool")
        if out.shape != out_decompressed.shape:
            raise ValueError("Output tensor shape does not match expected result shape")
        out.copy_(out_decompressed)
        return out
    else:
        return out_decompressed
