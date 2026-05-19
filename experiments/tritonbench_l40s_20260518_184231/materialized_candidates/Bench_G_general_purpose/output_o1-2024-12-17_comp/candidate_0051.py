import torch
import triton
import triton.language as tl


@triton.jit
def max_kernel_1(
    input_ptr,  # Pointer to input
    mid_ptr,    # Pointer to intermediate (partial maxima)
    n,          # Total number of elements in input
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.where(mask, tl.load(input_ptr + offsets), float("-inf"))
    max_val = tl.max(x, axis=0)
    tl.store(mid_ptr + pid, max_val)


@triton.jit
def max_kernel_2(
    mid_ptr,    # Pointer to partial maxima
    out_ptr,    # Pointer where final max is stored
    size_mid,   # Number of elements in mid
    BLOCK_SIZE: tl.constexpr
):
    # One block can reduce all 'size_mid' elements.
    # For simplicity, assume one block is enough to hold all partial maxima.
    # Otherwise, multiple passes would be required.
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size_mid
    partial_vals = tl.where(mask, tl.load(mid_ptr + offsets), float("-inf"))
    max_val = tl.max(partial_vals, axis=0)
    # Only let the first program_id store the final result
    if pid == 0:
        tl.store(out_ptr, max_val)


@triton.jit
def max_kernel(
    input_ptr,     # Pointer to the input tensor
    output_ptr,    # Pointer to where maximum values will be written
    index_ptr,     # Pointer to where indices of maxima will be written
    M,             # Number of blocks in the 'dim' dimension
    N,             # Size of the dimension being reduced
    K,             # Size of the rest of dimensions
    stride_in_m,   # Stride for the 'm' dimension in input
    stride_in_n,   # Stride for the reducing dimension in input
    stride_in_k,   # Stride for the 'k' dimension in input
    stride_out_m,  # Stride for the 'm' dimension in output
    stride_out_k,  # Stride for the 'k' dimension in output
    dim_size,      # Size of the dimension being reduced
    BLOCK_N: tl.constexpr
):
    # program_ids for (m, k)
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # Compute global coordinates
    # Each warp (pid_m, pid_k) will compute the max across the N-dimension
    offs_n = tl.arange(0, BLOCK_N)
    m_off = pid_m
    k_off = pid_k

    # Initialize max_val and max_idx
    max_val = float("-inf")
    max_idx = 0

    # Loop over dimension N in steps of BLOCK_N
    # (For simplicity, assume that N is a multiple of BLOCK_N or the last chunk is masked out.)
    for n_tile in range(0, N, BLOCK_N):
        n_offsets = n_tile + offs_n
        mask = n_offsets < N

        # Compute pointers to the input
        # input index = m_off * stride_in_m + k_off * stride_in_k + n_offsets * stride_in_n
        ptrs = input_ptr + (m_off * stride_in_m) + (k_off * stride_in_k) + (n_offsets * stride_in_n)
        vals = tl.where(mask, tl.load(ptrs), float("-inf"))

        # Compare to current maxima
        new_max = tl.maximum(max_val, vals)
        # We want the index that survived
        # If new_max != old_max, update index to current offsets
        better_mask = new_max > max_val
        max_val = new_max
        # If tie, keep old index (not strictly required if we only care about first occurrence)
        updated_idx = n_tile + offs_n
        max_idx = tl.where(better_mask, updated_idx, max_idx)

    # Store the results to output
    out_offset = (m_off * stride_out_m) + (k_off * stride_out_k)
    tl.store(output_ptr + out_offset, max_val)
    tl.store(index_ptr + out_offset, max_idx)


def max(input_tensor):
    """
    Computes the maximum value of a 1D tensor.
    Uses max_kernel_1 to compute partial maxima, then
    max_kernel_2 to reduce them into a final maximum.
    """
    n = input_tensor.numel()
    BLOCK_SIZE = 1024

    # Allocate intermediate and output buffers
    mid_shape = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    mid = torch.empty(mid_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    out = torch.empty(1, dtype=input_tensor.dtype, device=input_tensor.device)

    # Launch max_kernel_1
    grid_1 = lambda meta: ((mid_shape,),)
    max_kernel_1[grid_1](
        input_tensor, mid, n,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Launch max_kernel_2
    grid_2 = lambda meta: ((1,),)
    max_kernel_2[grid_2](
        mid, out, mid_shape,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out[0]


def max_dim(input_tensor, dim):
    """
    Computes the maximum values along a given dimension of the tensor and also
    returns the indices of those maxima.
    """
    # Validate dim
    rank = input_tensor.dim()
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        raise ValueError("Invalid dimension")

    # Permute the tensor so that 'dim' becomes the last dimension for simpler reduction
    # or rearrange the problem so that M is product of sizes before dim,
    # K is product of sizes after dim, and we reduce over the dimension 'dim_size'.
    shape = list(input_tensor.shape)
    dim_size = shape[dim]

    # Compute M = product of shapes up to dim
    M = 1
    for s in shape[:dim]:
        M *= s

    # Compute K = product of shapes after dim
    K = 1
    for s in shape[dim + 1 :]:
        K *= s

    # Flatten input into [M, dim_size, K]
    # The strides might need to be preserved for correct indexing
    # We'll gather them from the original tensor
    strides = input_tensor.stride()
    stride_in_m = strides[dim] if dim < rank - 1 else 1
    stride_in_n = strides[dim]
    stride_in_k = 1
    # To compute correct strides, we do a small re-mapping approach:
    # We'll create a contiguous sub-dim approach for demonstration, ignoring advanced broadcasting.
    # In a robust version, we should carefully handle strides. For demonstration, assume a contiguous layout.

    # Prepare output
    out_shape = shape[:dim] + shape[dim + 1 :]
    out_val = torch.empty(out_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    out_idx = torch.empty(out_shape, dtype=torch.int32, device=input_tensor.device)

    # Output strides for writing
    # For a 2D kernel indexing, we treat (m, k)
    # flatten out_val into MxK shape, same for out_idx
    stride_out_m = out_val.stride()[0] if len(out_val.shape) > 1 else 1
    stride_out_k = 1

    # Launch kernel in a 2D grid
    # grid = [M, K]
    BLOCK_N = 1024  # number of elements in the reduced dimension processed per program
    grid = (M, K)

    max_kernel[grid](
        input_tensor, out_val, out_idx,
        M, dim_size, K,
        stride_in_m, stride_in_n, stride_in_k,
        stride_out_m, stride_out_k, dim_size,
        BLOCK_N=BLOCK_N
    )

    return out_val, out_idx
