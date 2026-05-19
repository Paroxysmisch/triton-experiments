import torch
import triton
import triton.language as tl

# Max rank supported for simplicity
MAX_DIMS = 8

@triton.jit
def _tile_exp_kernel(
    in_ptr, out_ptr,
    in_stride0, in_stride1, in_stride2, in_stride3, in_stride4, in_stride5, in_stride6, in_stride7,
    out_size0, out_size1, out_size2, out_size3, out_size4, out_size5, out_size6, out_size7,
    in_size0, in_size1, in_size2, in_size3, in_size4, in_size5, in_size6, in_size7,
    rank: tl.constexpr,
    n_elems,
    BLOCK_SIZE: tl.constexpr
):
    # Compute the global range of indices this program will handle
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elems

    idx = offsets

    # Compute multi-dimensional coordinates in "out" via unraveling
    # For ranks < 8, we treat higher dims as size=1
    # and for stride = 0 to ensure correct indexing.
    coord7 = idx % out_size7
    idx = idx // out_size7
    coord6 = idx % out_size6
    idx = idx // out_size6
    coord5 = idx % out_size5
    idx = idx // out_size5
    coord4 = idx % out_size4
    idx = idx // out_size4
    coord3 = idx % out_size3
    idx = idx // out_size3
    coord2 = idx % out_size2
    idx = idx // out_size2
    coord1 = idx % out_size1
    idx = idx // out_size1
    coord0 = idx  # remaining

    # For each dimension, mod with input size
    # If rank <= d, we use zero. Otherwise, normal coordinate mod in_size
    in_coord0 = tl.where(rank > 0, coord0 % in_size0, 0)
    in_coord1 = tl.where(rank > 1, coord1 % in_size1, 0)
    in_coord2 = tl.where(rank > 2, coord2 % in_size2, 0)
    in_coord3 = tl.where(rank > 3, coord3 % in_size3, 0)
    in_coord4 = tl.where(rank > 4, coord4 % in_size4, 0)
    in_coord5 = tl.where(rank > 5, coord5 % in_size5, 0)
    in_coord6 = tl.where(rank > 6, coord6 % in_size6, 0)
    in_coord7 = tl.where(rank > 7, coord7 % in_size7, 0)

    # Compute input element index from the multi-dimensional coordinate
    in_index = (
        in_coord0 * in_stride0 +
        in_coord1 * in_stride1 +
        in_coord2 * in_stride2 +
        in_coord3 * in_stride3 +
        in_coord4 * in_stride4 +
        in_coord5 * in_stride5 +
        in_coord6 * in_stride6 +
        in_coord7 * in_stride7
    )

    # Load, apply exp, store to out
    val = tl.load(in_ptr + in_index, mask=mask, other=0.0)
    val = tl.exp(val)
    tl.store(out_ptr + offsets, val, mask=mask)


def fused_tile_exp(input, dims, *, out=None):
    # Ensure dims is a list
    if not isinstance(dims, (list, tuple)):
        raise TypeError("dims must be a list or tuple of ints")
    dims = list(dims)

    # Prepend ones if dims has fewer dimensions than input
    in_shape = list(input.shape)
    in_stride = list(input.stride())
    rank_in = len(in_shape)
    if len(dims) < rank_in:
        dims = [1] * (rank_in - len(dims)) + dims

    # Compute output shape
    out_shape = [in_shape[i] * dims[i] for i in range(rank_in)]

    # If out is provided, ensure it matches expected shape; otherwise allocate
    if out is not None:
        if list(out.shape) != out_shape:
            raise ValueError("out shape must match the tiled-then-exp result shape.")
        if out.device != input.device or out.dtype != input.dtype:
            raise ValueError("out must be on the same device and dtype as input.")
    else:
        out = torch.empty(out_shape, device=input.device, dtype=input.dtype)

    # Determine rank (capped at MAX_DIMS for this example)
    rank = min(rank_in, MAX_DIMS)

    # For dims beyond MAX_DIMS, raise an error
    if rank_in > MAX_DIMS:
        raise RuntimeError(f"Only up to {MAX_DIMS}-D tensors are supported in this example.")

    # Expand all sizes up to MAX_DIMS
