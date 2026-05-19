import math
import torch
import triton
import triton.language as tl


# ---------------------
# Helper Functions
# ---------------------
def can_use_int32_index(tensor: torch.Tensor) -> bool:
    """
    Checks if a tensor can use 32-bit indexing (i.e., the number
    of elements is within the range of 32-bit integers).
    """
    return tensor.numel() < 2**31


def dim_compress(tensor: torch.Tensor, dims: list) -> torch.Tensor:
    """
    A simple helper that permutes the specified dimensions (dims) 
    to the end of the tensor shape, effectively compressing them 
    into contiguous axes for easier calculations.
    """
    ndims = tensor.ndim
    dims = sorted(d % ndims for d in dims)
    # Move target dims to the end
    all_dims = list(range(ndims))
    for d in reversed(dims):
        all_dims.append(all_dims.pop(d))
    # Permute and reshape
    permuted = tensor.permute(all_dims)
    # Calculate the product of the moved dims
    product = 1
    for d in dims:
        product *= tensor.shape[d]
    # Reshape so that the moved dims are flattened into one axis
    shape = list(permuted.shape)
    new_shape = shape[: ndims - len(dims)] + [product]
    return permuted.reshape(new_shape)


def cfggen():
    """
    Helper function to generate configurations for Triton autotuning.
    """
    block_m = [1, 2, 4, 8]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": 1024}, num_warps=4) for m in block_m
    ]
    return configs


# ---------------------
# Kernel 1: max_kernel_1
# ---------------------
@triton.jit
def max_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    """
    This kernel computes block-level maximum values in a 1D tensor.
    'pid' is the program id. Each block processes BLOCK_SIZE elements
    starting at offset = pid * BLOCK_SIZE. The max value for each block 
    is stored in 'mid'.
    """
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    block_max = tl.max(val)
    tl.store(mid + pid, block_max)


# ---------------------
# Kernel 2: max_kernel_2
# ---------------------
@triton.jit
def max_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    """
    This kernel takes the intermediate results from 'mid',
    computes their maximum, and stores the result in 'out'.
    """
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
    block_max = tl.max(mid_val)
    tl.store(out, block_max)


# ---------------------
# Kernel 3: max_kernel
# ---------------------
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def max_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    """
    Generic maximum reduction kernel. Each program id (pid) handles
    one block of rows. For each row in a block, it scans along N
    columns and computes the maximum. Results are stored in 'out'.
    """
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)

    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)[:, None]  # (BLOCK_M, 1)
    row_mask = rows < M

    # Advance the pointer for the input/output
    inp += rows * N
    out += rows

    # Initialize a buffer to store per-thread block maxima
    max_vals = tl.full([BLOCK_M, BLOCK_N], -float("inf"), dtype=tl.float32)

    # Loop over columns in BLOCK_N chunks
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        val = tl.load(inp + cols, mask=mask, other=-float("inf"))
        val_float32 = val.to(tl.float32)
        max_vals = tl.maximum(max_vals, val_float32)

    # Compute the max over each row
    row_max = tl.max(max_vals, axis=1)[:, None]
    tl.store(out, row_max, mask=row_mask)


# ---------------------
# Wrapper: max
# ---------------------
def max(inp: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    """
    Computes the maximum value across the entire input tensor.
    Uses max_kernel_1 and max_kernel_2. The final result is written 
    to 'out'. The shape of 'out' depends on 'keepdim'.
    """
    M = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = (M + block_size - 1) // block_size  # triton.cdiv
    block_mid = triton.next_power_of_2(mid_size)

    use_int64_index = not can_use_int32_index(inp)
    dtype = inp.dtype
    device = inp.device

    # Intermediate storage
    mid = torch.empty(mid_size, dtype=dtype, device=device)

    # Output storage
    if keepdim:
        out_shape = [1] * inp.dim()
        out = torch.empty(out_shape, dtype=dtype, device=device)
    else:
        out = torch.empty([], dtype=dtype, device=device)

    # Launch first kernel
    max_kernel_1[(mid_size,)](
        inp, mid, M, block_size, INT64_INDEX=use_int64_index
    )
    # Launch second kernel
    max_kernel_2[(1,)](
        mid, out, mid_size, block_mid
    )
    return out


# ---------------------
# Extended: max_dim
# ---------------------
def max_dim(inp: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """
    Computes the maximum value of 'inp' along a specified 'dim'.
    This function internally performs a reduction along 'dim' via
    the 'max_kernel'. The output is then reshaped based on 'keepdim'.
    """
    # Validate & wrap dimension value
    ndims = inp.ndim
    if dim < 0:
        dim += ndims
    assert 0 <= dim < ndims, f"Invalid dim={dim} for tensor of rank {ndims}"

    # Prepare compressed version of the input
    # to reduce along the last dimension
    shape = list(inp.shape)
    dims = [dim]
    compressed = dim_compress(inp, dims)

    # M is the product of the dimensions except the reduced one,
    # N is the size of the reduced dimension.
    N = shape[dim]
    M = compressed.numel() // N

    # Output shape after reduction
    out_shape = shape.copy()
    out_shape[dim] = 1

    # Create output buffer
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)

    # Decide on 64-bit indexing
    use_int64_index = not can_use_int32_index(compressed)

    # Set up Triton grid
    def grid(meta):
        return ( (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"], )

    # Launch the kernel
    max_kernel[grid](
        compressed,
        out,
        M,
        N,
        INT64_INDEX=use_int64_index
    )

    # Squeeze if keepdim is False
    if not keepdim:
        out = out.squeeze(dim=dim)

    return out
