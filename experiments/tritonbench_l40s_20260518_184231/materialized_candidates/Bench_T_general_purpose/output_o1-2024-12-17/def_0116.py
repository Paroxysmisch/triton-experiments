import torch
import triton
import triton.language as tl

@triton.jit
def _sum_dim_kernel(
    input_ptr, output_ptr,
    row_size,  # number of rows
    col_size,  # number of columns to sum over
    stride_in_row, stride_in_col,
    stride_out,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    # Each program handles one row
    if row_id >= row_size:
        return

    # Start pointer for this row
    row_input_offset = row_id * stride_in_row
    # Accumulator
    acc = 0.0

    # Loop over the columns in steps of BLOCK_SIZE
    for start_col in range(0, col_size, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets + start_col < col_size
        ptrs = input_ptr + row_input_offset + (offsets + start_col) * stride_in_col
        vals = tl.load(ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)

    # Store result
    tl.store(output_ptr + row_id * stride_out, acc)

def sum(input, dim, keepdim=False, *, dtype=None):
    # Convert dim to a tuple if it's not already
    if isinstance(dim, int):
        dim = (dim,)

    # Handle multiple dims by reducing one at a time
    out = input
    for d in sorted(dim, reverse=True):
        # Move target dim to the last dimension for easy kernel call
        order = list(range(out.ndim))
        order[d], order[-1] = order[-1], order[d]
        out = out.permute(order)

        # Shape info
        row_size = 1
        for i in range(out.ndim - 1):
            row_size *= out.shape[i]
        col_size = out.shape[-1]

        # Prepare output shape for this reduction
        new_shape = list(out.shape[:-1])
        new_shape[-1] = 1 if keepdim else 1  # We'll unsqueeze if keepdim=True later
        out_buf = torch.empty(new_shape, dtype=out.dtype, device=out.device)

        # Strides for input
        stride_in_row = out.stride(0)
        for i in range(1, out.ndim - 1):
            stride_in_row *= out.shape[i]
        stride_in_col = out.stride(-1)

        # Strides for output
        stride_out = out_buf.stride(0)
        for i in range(1, len(new_shape)):
            stride_out *= out_buf.shape[i]

        # Launch kernel
        grid = (row_size,)
        BLOCK_SIZE = 128
        triton.run(
            _sum_dim_kernel,
            grid=grid,
            num_warps=4,
            BLOCK_SIZE=BLOCK_SIZE,
            inputs=[
                out.data_ptr(), out_buf.data_ptr(),
                row_size, col_size,
                stride_in_row, stride_in_col,
                stride_out
            ]
        )

        # Reshape/permute back
        # If keepdim=True, we keep the reduced dimension size=1
        # otherwise we remove that dimension
        if keepdim:
            out = out_buf
        else:
            out = out_buf.squeeze(-1)

        # Undo the permute
        reverse_perm = list(range(len(order)))
        for i, o in enumerate(order):
            reverse_perm[o] = i
        out = out.permute(reverse_perm)

    # Cast to desired dtype if provided
    if dtype is not None:
        out = out.to(dtype)

    return out
