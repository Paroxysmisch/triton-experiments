n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    tl.debug_barrier()

    square_output = row * row
    mean_output = tl.sum(square_output)/n_cols + eps
    rms_output = tl.sqrt(mean_output)
    
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride # TODO: optimization: always 1 after the reduction
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, rms_output, mask=col_offsets < n_cols)

@triton.jit
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
    mean_output = tl.sum(square_output)/n_cols + eps
    rms_output = tl.sqrt(mean_output)
    normalized_output = row/rms_output
    
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride # TODO: optimization: always 1 after the reduction
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, normalized_output, mask=col_offsets < n_cols)

@triton.jit
def gelu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # The rows are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * BLOCK_SIZE
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_elements, other=-float('inf'))
    tl.debug_barrier()

    # GELU = X * 0.5 * (1.0 + tanh[(sqrt(2/pi) * X * (1 + 0.044715 * X^2))])
    sqrt_2_over_pi = 0.7978845608028654
    tanh_arg = sqrt_2_over_pi * row * (1 + 0.044715 * row * row)
    gelu_output = row * 0.5 * (1.0 + tl.tanh(tanh_arg))
    
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * BLOCK_SIZE
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, gelu_output, mask=col_offsets < n_elements)

@triton.jit
def dropout_kernel(input_ptr, output_ptr, n_elements, p, seed, offset, BLOCK_SIZE: tl.constexpr):
    # The rows are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * BLOCK_SIZE
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_elements, other=-float('inf'))
    tl.debug_barrier()

    # dropout with random mask
    random = tl.rand(seed, offset, row_idx)
    mask = random > p
    dropout_output = tl.where(mask, row/p, 0.0)
    
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * BLOCK_SIZE
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, dropout_output, mask=col_offsets < n_elements)

@triton.jit
def subtract_kernel(output_ptr, input1_ptr, input2_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # The rows are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input1_ptr + row_idx * BLOCK_SIZE
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input1_ptrs = row_start_ptr + col_offsets
    input2_ptrs = input2_ptr + row_idx * BLOCK_SIZE + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    input1 = tl.load(input1_ptrs, mask=col_offsets < n_elements, other=-float('inf'))
    input2 = tl.load(input2_ptrs, mask=col_offsets < n_elements, other=-float('inf'))
    tl.debug_barrier()

    # Subtract
    subtract_output = input1 - input2
    
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * BLOCK_SIZE
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, subtract_output, mask=col_offsets < n_elements)

def fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-5, *, out=None):
    if approximate != 'none':
        raise NotImplementedError(f'Only \'none\' is supported for approximate
