def rms_norm(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    # The rows are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    tl.debug_barrier()

    square_output = row * row
    rms = tl.sqrt(tl.sum(square_output)/n_cols + eps)
    
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride # TODO: optimization: always 1 after the reduction
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, rms, mask=col_offsets < n_cols)

def square(x):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    assert output.is_contiguous()
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # TODO: tune these
    num_warps = 8
    # Launch the Triton kernel
    square_kernel[(n_rows,)](
        output,
        x,
        x.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )
    return output

def mean_of_squares(x, eps=1e-6):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    assert output.is_contiguous()
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # TODO: tune these
    num_warps = 8
    # Launch the Triton kernel
    mean_of_squares_kernel[(n_rows,)](
        output,
        x,
        x.stride(0),
        output.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )
    return output

def rms(x, eps=1e-6):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    assert output.is_contiguous()
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # TODO: tune these
    num_warps = 8
    # Launch the Triton kernel
    rms_kernel[(n_rows,)](
        output,
        x,
        x.stride(0),
        output.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )
    return output

def rms_norm(x, eps=1e-6):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    assert output.is_contiguous()
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # TODO: tune these
    num_warps = 8
    # Launch the Triton kernel
    rms_norm[(n_rows,)](
        output,
        x,
        x.stride(0),
        output.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )
    return output
