import torch
import triton
import triton.language as tl

@triton.jit
def _lp_norm_kernel(
    x_ptr,  # [num_rows, row_size]
    norms_ptr,  # [num_rows]
    n_rows,  # total rows along outer dimension
    row_size,  # total size along reduced dimension
    stride_row,  # stride along outer dimension
    stride_col,  # stride along reduced dimension
    p,  # p for Lp norm
    eps,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    # Each block processes one row
    # If row_id >= n_rows, do nothing
    if row_id >= n_rows:
        return

    # Base pointers
    row_start = x_ptr + row_id * stride_row

    # Indices for a block's partial sum
    block_start = tl.arange(0, BLOCK_SIZE)
    offsets = row_start + block_start * stride_col

    # Running partial sum in fp32
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Loop over row_size in steps of BLOCK_SIZE
    for start in range(0, row_size, BLOCK_SIZE):
        mask = block_start + start < row_size
        x = tl.load(offsets + start * stride_col, mask=mask, other=0.0)
        # |x|^p
        x_abs_p = tl.abs(x) ** p
        acc += tl.where(mask, x_abs_p, 0.0)

    # Final sum across thread block
    total = tl.sum(acc, axis=0)
    # Write back final norm (p-root)
    norm_val = tl.float32_pow(total, 1.0 / p)
    norm_val = tl.where(norm_val < eps, eps, norm_val)
    tl.store(norms_ptr + row_id, norm_val)


@triton.jit
def _normalize_kernel(
    x_ptr,      # [num_rows, row_size]
    norms_ptr,  # [num_rows]
    out_ptr,    # [num_rows, row_size]
    n_rows,
    row_size,
    stride_row_in,
    stride_col_in,
    stride_row_out,
    stride_col_out,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    if row_id >= n_rows:
        return

    norm_val = tl.load(norms_ptr + row_id)
    x_row_start = x_ptr + row_id * stride_row_in
    out_row_start = out_ptr + row_id * stride_row_out

    block_idx = tl.arange(0, BLOCK_SIZE)
    for start in range(0, row_size, BLOCK_SIZE):
        mask = block_idx + start < row_size
        x_val = tl.load(x_row_start + (block_idx + start) * stride_col_in, mask=mask, other=0.0)
        # normalize
        normed = x_val / norm_val
        tl.store(out_row_start + (block_idx + start) * stride_col_out, normed, mask=mask)


@triton.jit
def _dot_kernel(
    x1_ptr,      # [num_rows, row_size]
    x2_ptr,      # [num_rows, row_size]
    out_ptr,     # [num_rows]
    n_rows,
    row_size,
    stride_row_x1,
    stride_col_x1,
    stride_row_x2,
    stride_col_x2,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    if row_id >= n_rows:
        return

    x1_row_start = x1_ptr + row_id * stride_row_x1
    x2_row_start = x2_ptr + row_id * stride_row_x2

    block_idx = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for start in range(0, row_size, BLOCK_SIZE):
        mask = block_idx + start < row_size
        a = tl.load(x1_row_start + (block_idx + start) * stride_col_x1, mask=mask, other=0.0)
        b = tl.load(x2_row_start + (block_idx + start) * stride_col_x2, mask=mask, other=0.0)
        acc += tl.where(mask, a * b, 0.0)

    dot_val = tl.sum(acc, axis=0)
    tl.store(out_ptr + row_id, dot_val)


@triton.jit
def _lp_norm_kernel2(
    x_ptr,      # [num_rows, row_size]
    out_ptr,    # [num_rows]
    n_rows,
    row_size,
    stride_row,
    stride_col,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    if row_id >= n_rows:
        return

    row_start = x_ptr + row_id * stride_row
    block_start = tl.arange(0, BLOCK_SIZE)
    offsets = row_start + block_start * stride_col
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for start in range(0, row_size, BLOCK_SIZE):
        mask = block_start + start < row_size
        x = tl.load(offsets + start * stride_col, mask=mask, other=0.0)
        x_sq = x * x
        acc += tl.where(mask, x_sq, 0.0)

    total = tl.sum(acc, axis=0)
    norm_val = tl.sqrt(total)
    norm_val = tl.where(norm_val < eps, eps, norm_val)
    tl.store(out_ptr + row_id, norm_val)


def normalized_cosine_similarity(
    x1: torch.Tensor,
    x2: torch.Tensor,
    dim: int = 1,
    eps_similarity: float = 1e-8,
    p_norm: float = 2.0,
    eps_norm: float = 1e-12
) -> torch.Tensor:
    # Broadcast x2 to x1 if needed
    x2 = x2.broadcast_to(x1.shape)

    # Move both to CUDA if not already
    x1_dev = x1.contiguous().to('cuda')
    x2_dev = x2.contiguous().to('cuda')

    # Permute so that dim=1 is the reduced dimension (row_size),
    # i.e., shape = [N, M] with M as the dimension to reduce
    # for general dim: reorder to (dim -> 1)
    # For simplicity, assume x1, x2 are at least 2D
    dims = list(range(x1_dev.ndim))
    if dim != 1:
        dims[1], dims[dim] = dims[dim], dims[1]
        x1_dev = x1_dev.permute(dims)
        x2_dev = x2_dev.permute(dims)

    N = x1_dev.shape[0]
    M = x1_dev.shape[1]
    # Flatten all trailing dims after 2
    trailing_shape = x1_dev.shape[2:]
    trailing_size = 1
    for s in trailing_shape:
        trailing_size *= s

    # Reshape to [N * trailing_size, M] to handle all trailing in blocks
    x1_dev = x1_dev.reshape(N * trailing_size, M)
    x2_dev = x2_dev.reshape(N * trailing_size, M)
    out_normed_x1 = torch.empty_like(x1_dev)
    out_normed_x2 = torch.empty_like(x2_dev)

    # Compute Lp norms of x1 and x2
    norms_x1 = torch.empty((x1_dev
