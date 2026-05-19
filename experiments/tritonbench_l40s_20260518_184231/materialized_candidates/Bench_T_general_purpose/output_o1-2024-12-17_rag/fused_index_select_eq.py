import torch
import triton
import triton.language as tl

def dim_compress(tensor, dim):
    """
    Moves the dimension 'dim' to the last dimension, then returns a contiguous view.
    This helps in reducing the multi-dimensional index select problem to 2D.
    """
    if dim < 0:
        dim += tensor.ndim
    permute_order = list(range(tensor.ndim))
    permute_order.append(permute_order.pop(dim))
    tensor = tensor.permute(permute_order).contiguous()
    return tensor

@triton.jit
def fused_index_select_eq_kernel(
    inp_ptr,                  # input tensor
    out_ptr,                  # output tensor (boolean result stored as 0/1)
    M,                        # number of rows
    N,                        # number of columns
    idx_ptr,                  # index tensor
    idx_len,                  # length of index
    other_ptr,                # pointer to 'other' if it's a tensor
    other_scalar,             # scalar value of 'other' if it's a float
    use_scalar: tl.constexpr, # flag indicating if 'other' is scalar
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    
    rows_offsets = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols_offsets = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)
    
    rows_mask = rows_offsets < M
    cols_mask = cols_offsets < N
    block_mask = rows_mask & cols_mask
    out_mask = rows_mask & (cols_offsets < idx_len)

    indices = tl.load(idx_ptr + cols_offsets, mask=(cols_offsets < idx_len), other=0)
    inp_off = rows_offsets * N + indices[None, :]
    out_off = rows_offsets * idx_len + cols_offsets[None, :]

    selected = tl.load(inp_ptr + inp_off, mask=block_mask, other=0.0)
    
    # Load 'other' either as scalar or from pointer
    if use_scalar:
        other_val = other_scalar
    else:
        # Load 'other' from the same offsets as out_off, expecting same shape
        other_val = tl.load(other_ptr + out_off, mask=block_mask, other=0.0)
    
    # Compute equality
    eq_mask = selected == other_val
    out_val = tl.where(eq_mask, 1, 0)
    tl.store(out_ptr + out_off, out_val, mask=out_mask)

def fused_index_select_eq(input, dim, index, other, *, out=None):
    """
    fused_index_select_eq(input, dim, index, other, *, out=None) -> Tensor
    
    Selects elements from 'input' along dimension 'dim' using 'index', then
    compares them against 'other' element-wise for equality. Returns a boolean
    tensor of the same shape as the selected elements.
    """
    assert dim >= -input.ndim and dim < input.ndim, "Invalid dim."
    assert index.ndim <= 1, "Index should be at most 1D."
    assert (index >= 0).all() and (index < input.size(dim)).all(), "Index out of range."

    if index.ndim == 0:
        index = index.unsqueeze(0)
    dim = dim % input.ndim
    input_shape = list(input.shape)
    index_len = index.numel()

    # Compress input along the dimension to 2D
    inp_compressed = dim_compress(input, dim)
    N = input_shape[dim]
    M = inp_compressed.numel() // N

    # Prepare the output tensor
    out_shape = list(inp_compressed.shape)
    out_shape[-1] = index_len
    if out is None:
        # Store result as bool
        out_tensor = torch.empty(out_shape, dtype=torch.int32, device=input.device)
    else:
        # Ensure out has correct shape and dtype
        assert out.shape == tuple(out_shape), "Output shape mismatch."
        assert out.dtype == torch.int32, "Output dtype must be int32 for storing 0/1."
        out_tensor = out

    # Handle 'other'
    use_scalar = False
    other_scalar = 0.0
    if isinstance(other, (int, float)):
        use_scalar = True
        other_scalar = float(other)
        other_ptr = None
    else:
        # Broadcast 'other' to the selected shape if needed
        # Make final shape match [M, index_len] after compression
        target_shape = (M, index_len)
        other_expanded = other.expand(target_shape).contiguous()
        other_ptr = other_expanded.view(-1)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(index_len, meta["BLOCK_N"]),
    )

    fused_index_select_eq_kernel[grid](
        inp_compressed,                        # inp_ptr
        out_tensor,                            # out_ptr
        M,                                     # M
        N,                                     # N
        index,                                 # idx_ptr
        index_len,                             # idx_len
        other_ptr if not use_scalar else None, # other_ptr
        other_scalar,                          # other_scalar
        use_scalar,                            # use_scalar
    )

    # Convert the int32 result to bool
    out_tensor_bool = out_tensor != 0

    # If compression changed the dimension order, revert
    if dim != (out_tensor_bool.ndim - 1):
        order = list(range(out_tensor_bool.ndim - 1))
        order.insert(dim, out_tensor_bool.ndim - 1)
        out_tensor_bool = out_tensor_bool.permute(order).contiguous()

    return out_tensor_bool
