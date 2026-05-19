import math
import torch
import triton
import triton.language as tl


def can_use_int32_index(tensor: torch.Tensor) -> bool:
    # Checks if the total number of elements can fit into a 32-bit int
    return tensor.numel() < 2**31


def dim_compress(inp: torch.Tensor, dims):
    # Sort dims to handle them in ascending order
    dims = sorted(dims)
    shape = list(inp.shape)
    # All dims to compress into a single dimension
    # Move them so they are contiguous
    new_shape = []
    compressed_size = 1
    for i, s in enumerate(shape):
        if i in dims:
            compressed_size *= s
        else:
            new_shape.append(s)
    new_shape.insert(dims[0], compressed_size)
    return inp.reshape(new_shape)


@triton.jit
def argmax_kernel_1(
    inp_ptr,            # *F32
    mid_value_ptr,      # *F32
    mid_index_ptr,      # *I64 or *I32 (decided by INT64_INDEX)
    M,
    BLOCK_SIZE: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M

    # Load values
    vals = tl.load(inp_ptr + offset, mask=mask, other=-float("inf"))
    # Compute maximum value
    val_max = tl.max(vals)
    # Find the index for that maximum (may pick the last occurrence if multiple maxima)
    eq_mask = vals == val_max
    idx_contrib = offset.to(tl.int64) * eq_mask.to(tl.int64)
    idx_max = tl.max(idx_contrib)

    # Store into mid arrays
    tl.store(mid_value_ptr + pid, val_max)
    tl.store(mid_index_ptr + pid, idx_max)


@triton.jit
def argmax_kernel_2(
    mid_value_ptr,      # *F32
    mid_index_ptr,      # *I64 or *I32
    out_ptr,            # *I64 or *I32
    mid_size,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size

    vals = tl.load(mid_value_ptr + offset, mask=mask, other=-float("inf"))
    val_max = tl.max(vals)

    eq_mask = vals == val_max
    idx_contrib = offset.to(tl.int64) * eq_mask.to(tl.int64)
    idx_max = tl.max(idx_contrib)

    # Read the real index
    real_idx = tl.load(mid_index_ptr + idx_max)
    # Final argmax
    tl.store(out_ptr, real_idx)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=4),
    ],
    key=["M", "N"],
)
@triton.jit
def argmax_kernel(
    inp_ptr,        # *F32
    out_ptr,        # *I64 or *I32
    M,              # total rows
    N,              # reduction dimension
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = rows < M

    # Pointers for input & output
    inp_offset = rows * N
    inp_ptr = inp_ptr + inp_offset
    out_ptr = out_ptr + rows

    # Initialize
    max_vals = tl.full([BLOCK_M, 1], -float("inf"), tl.float32)
    max_idxs = tl.zeros([BLOCK_M, 1], tl.int64)

    # Traverse over the dimension to reduce
    for offs in range(0, N, BLOCK_N):
        cols = offs + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask

        # Load data
        vals = tl.load(inp_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        # Compare with current max
        greater_mask = vals > max_vals
        # Update max values
        max_vals = tl.where(greater_mask, vals, max_vals)
        # Compute new potential indices
        new_idx = cols.to(tl.int64)
        max_idxs = tl.where(greater_mask, new_idx, max_idxs)

    # Store final indices (argmax)
    # Need to store 64 or 32 bit indices
    if INT64_INDEX:
        tl.store(out_ptr, max_idxs.to(tl.int64), mask=row_mask)
    else:
        tl.store(out_ptr, max_idxs.to(tl.int32), mask=row_mask)


def argmax(inp: torch.Tensor, dim=None, keepdim=False):
    # Decide index dtype
    use_int64_index = not can_use_int32_index(inp)
    index_dtype = torch.int64 if use_int64_index else torch.int32

    if dim is None:
        M = inp.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)

        mid_value = torch.empty((mid_size,), dtype=inp.dtype, device=inp.device)
        mid_index = torch.empty((mid_size,), dtype=index_dtype, device=inp.device)

        if not keepdim:
            out_shape = []
        else:
            out_shape = [1 for _ in range(inp.dim())]

        out = torch.empty(out_shape, dtype=index_dtype, device=inp.device)

        # Launch first stage
        argmax_kernel_1[(mid_size,)](
            inp, mid_value, mid_index, M,
            block_size,
            INT64_INDEX=use_int64_index
        )
        # Launch second stage
        argmax_kernel_2[(1,)](
            mid_value, mid_index, out, mid_size, block_mid
        )
