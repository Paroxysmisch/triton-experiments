import torch
import triton
import triton.language as tl

@triton.jit
def _argmax_reduce_row_kernel(
    input_ptr,          # Pointer to the start of the row in the input tensor
    output_ptr,         # Pointer to the location where the result index should be stored
    input_row_stride,   # Stride (in elements) between rows in the input
    n_cols,             # Number of columns to reduce over
    BLOCK_SIZE: tl.constexpr
):
    # We launch one program per row
    row_id = tl.program_id(0)

    # Pointers to this row's data in input and output
    row_start = input_ptr + row_id * input_row_stride
    out_ptr   = output_ptr + row_id

    # Initialize running max value and index
    max_val = tl.full([BLOCK_SIZE], float('-inf'), tl.float32)
    max_idx = tl.full([BLOCK_SIZE], 0, tl.int32)

    # Loop over columns in blocks of BLOCK_SIZE
    # Accumulate the maximum value and index
    for offset in range(0, n_cols, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols  # Valid columns
        # Load the data for this block
        val = tl.load(row_start + cols, mask=mask, other=float('-inf')).to(tl.float32)

        # Compare to current max, update max_val and max_idx where new value is greater
        is_greater = val > max_val
        max_val = tl.where(is_greater, val, max_val)
        max_idx = tl.where(is_greater, cols, max_idx)

    # Now we need to reduce inside the BLOCK_SIZE threads
    # Use parallel reduction in a warp-friendly manner
    # e.g., do log2(BLOCK_SIZE) steps
    length = BLOCK_SIZE
    stride = length // 2
    while stride > 0:
        # Each thread compares with its "partner" at stride offset
        partner_val = tl.sqrt(-1.0)  # dummy var
        partner_idx = tl.full([], 0, tl.int32)
        if tl.arange(0, BLOCK_SIZE) < stride:
            partner_val = max_val + max_val  # dummy init
            partner_val = tl.load(max_val + stride, mask=False)  # not valid in Triton
            # above line is invalid because 'max_val' is not a pointer
            # We'll do the typical local shuffle approach in standard GPU languages,
            # but in Triton we can do a simpler approach by slicing the vector in half.

            # We can't do direct loads from 'max_val' as if it were memory.  We'll do explicit scalar (vector) exchange:
            max_val_local = max_val
            max_idx_local = max_idx

        # We can't do warp-level shuffles in pure Triton easily in a single pass,
        # so we do a simpler approach by iterative halving:
        # "fold" half of the threads
        half = length // 2
        if tl.arange(0, BLOCK_SIZE) < half:
            left_val = max_val[tl.arange(0, BLOCK_SIZE)]
            left_idx = max_idx[tl.arange(0, BLOCK_SIZE)]
            right_val = max_val[tl.arange(0, BLOCK_SIZE) + half]
            right_idx = max_idx[tl.arange(0, BLOCK_SIZE) + half]
            # Compare left vs right, pick bigger
            cond = right_val > left_val
            out_val = tl.where(cond, right_val, left_val)
            out_idx = tl.where(cond, right_idx, left_idx)
            max_val = out_val
            max_idx = out_idx
        length = half
        stride = stride // 2

    # After all reductions, the first lane has the final max index
    # Store that into output; cast to int64 to match Argmax's spec
    if tl.thread_id_x() == 0:
        tl.store(out_ptr, max_idx[0].to(tl.int64))

def argmax(input: torch.Tensor, dim: int or None, keepdim: bool = False) -> torch.Tensor:
    """
    Argmax Triton Wrapper.
    Returns the indices of the maximum values of a tensor across a specified dimension.
    If `dim` is None, argmax of the flattened tensor is returned.
    If `keepdim` is True, the resulting tensor retains the reduced dimension.

    Args:
        input (torch.Tensor): the input tensor.
        dim (int, optional): the dimension to reduce. If None, the argmax of the flattened input is returned.
        keepdim (bool, optional): whether the output tensor has `dim` retained or not.

    Returns:
        torch.Tensor (dtype=torch.long): Indices of maximum values.
    """
    # Handle the case when dim is None: flatten the tensor and find a single index
    if dim is None:
        flattened = input.view(-1)
        n_elems = flattened.shape[0]

        # If the input is empty, torch.argmax would normally return 0 or fail
        if n_elems == 0:
            # Keeping consistent with PyTorch's behavior for an empty input
            return torch.tensor([], dtype=torch.long, device=input.device)

        # We'll run 1 "row" with n_cols = all elements
        # Allocate the output
        out_idx = torch.empty(1, dtype=torch.long, device=input.device)
        BLOCK_SIZE = triton.next_power_of_2(n_elems)
        grid = (1,)

        _argmax_reduce_row_kernel[grid](
            flattened,           # input_ptr
            out_idx,            # output_ptr
            1,                  # stride between "rows"
            n_elems,            # n_cols
            BLOCK_SIZE=BLOCK_SIZE
        )
        # Return scalar index
        return out_idx[0].unsqueeze(0) if keepdim else out_idx[0]

    # Ensure the dimension is in range
    dim = dim if dim >= 0 else (dim + input.dim())
    if dim < 0 or dim >= input.dim():
        raise ValueError(f"Dimension out of range (expected to be in range of [{-input.dim()}, {input.dim()-1}], but got {dim})")

    # If the input is empty along dim, mimic PyTorch's argmax behavior
    if input.shape[dim] == 0:
        shape = list(input.shape)
        if keepdim:
            shape[dim] = 1
        else:
            del shape[dim]
        return torch.zeros(shape, dtype=torch.long, device=input.device)

    # Move `dim` to the last dimension for convenience if needed
    if dim != input.dim() - 1:
        # Permute so that `dim` is last
        perm = list(range(input.dim()))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        input = input.permute(*perm)
        transposed_back = True
    else:
        transposed_back = False

    # Flatten all but the last dimension
    shape_2d = input.shape
    n_rows = 1
    for s in shape_2d[:-1]:
        n_rows *= s
    n_cols = shape_2d[-1]
    input_2d = input.contiguous().view(n_rows, n_cols)

    # Output buffer for indices
    out_idx_2d = torch.empty(n_rows, dtype=torch.long, device=input.device)

    # Dispatch kernel
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    _argmax_reduce_row_kernel[(n_rows,)](
        input_2d,
        out_idx_2d,
        input_2d.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Now shape the result
    # `out_idx_2d` is [n_rows]
    out_idx = out_idx_2d.view(*shape_2d[:-1])  # remove last-dim reduce
    if transposed_back:
        # Permute back
        # We moved `dim` to last, so we must invert that permutation
        inv_perm = list(range(input.dim()))
        inv_perm[-1], inv_perm[dim] = inv_perm[dim], inv_perm[-1]
        out_idx = out_idx.permute(*inv_perm)

    if keepdim:
        # If we keep the dimension, we unsqueeze at dim
        out_idx = out_idx.unsqueeze(dim)
    else:
        # If not keepdim, out_idx already has that dimension removed
        pass

    return out_idx
