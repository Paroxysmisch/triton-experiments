import torch
import triton
import triton.language as tl

# --- TRITON LOG SOFTMAX ----
@triton.jit
def log_softmax_kernel(input_pointer,
                       out_pointer,
                       input_row_stride,
                       out_row_stride,
                       n_cols,
                       BLOCK_SIZE: tl.constexpr,
                       ):
    # the rows of the log softmax are independent
    # so we parallelize across those
    row_idx = tl.program_id(0)

    # stride is how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_pointer + (row_idx * input_row_stride)

    # Each thread within the block will handle a different element of the row.
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets  # Calculate pointers for each element in the current row.

    # Load the current row data from memory
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    row_max = tl.max(row, axis=0)  # Find the maximum value in the row to avoid overflow
    row_minus_max = row - row_max  # Normalize by the max

    # Compute the log sum of exponentials
    log_sum_exp = tl.log(tl.sum(tl.exp(row_minus_max), axis=0))

    # Compute the log softmax
    log_softmax_output = row_minus_max - log_sum_exp

    # Write the results to memory
    output_row_start_ptr = out_pointer + row_idx * out_row_stride  # Calculate the start pointer of the current row in the output data.
    output_ptrs = output_row_start_ptr + col_offsets  # Calculate pointers for each element in the output row
    tl.store(output_ptrs, log_softmax_output, mask=col_offsets < n_cols)  # Store the log softmax results in the appropriate locations in GPU memory

import torch
import triton
import triton.language as tl

class LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        # Save the input and dimension for the backward pass
        ctx.save_for_backward(x)
        ctx.dim = dim

        n_rows, n_cols = x.shape
        BLOCK_SIZE, num_warps = calculate_settings_a(n_cols)

        # Allocate output
        y = torch.empty_like(x)

        # Launch the kernel with calculated settings
        log_softmax_kernel[(n_rows,)](
            input_pointer=x,
            out_pointer=y,
            input_row_stride=x.stride(0),
            out_row_stride=y.stride(0),
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_wars
        )
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        dim = ctx.dim

        n_rows, n_cols = x.shape
        BLOCK_SIZE, num_warps = calculate_settings_a(n_cols)

        # Allocate output for the gradient
        grad_input = torch.empty_like(x)

        # Launch the backward kernel with calculated settings
        log_softmax_backward_kernel[(n_rows,)](
            grad_output,
            x,
            grad_input,
            grad_output.stride(0),
            x.stride(0),
            grad_input.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )
        return grad_input, None

@triton.jit
def log_softmax_backward_kernel(grad_output_pointer,
                                input_pointer,
                                grad_input_pointer,
                                grad_output_row_stride,
                                input_row_stride,
                                grad_input_row_stride,
                                n_cols,
                                BLOCK_SIZE: tl.constexpr,
                                ):
    # the rows of the log softmax are independent
    # so we parallelize across those
    row_idx = tl.program_id(0)

    # stride is how much we need to increase the pointer to advance 1 row
    grad_output_row_start_ptr = grad_output_pointer + (row_idx * grad_output_row_stride)
    input_row_start_ptr = input_pointer + (row_idx * input_row_stride)
    grad_input_row_start_ptr = grad_input_pointer + (row_idx * grad_input_row_stride)

    # Each thread within the block will handle a different element of the row.
    col_offsets = tl.arange(0, BLOCK_SIZE)
    grad_output_ptrs = grad_output_row_start_ptr + col_offsets
    input_ptrs = input_row_start_ptr + col_offsets
    grad_input_ptrs = grad_input_row_start_ptr + col_offsets

    # Load the current row data from memory
    grad_output_row = tl.load(grad_output_ptrs, mask=col_offsets < n_cols, other=0.0)
    input_row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=0.0)

    # Compute the gradient of the log softmax
    grad_input_row = grad_output_row - tl.sum(grad_output_row * tl.exp(input_row), axis=0)

    # Write the results to memory
    tl.store(grad_input_ptrs, grad_input_row, mask=col_offsets < n_cols)

def log_softmax(x, dim, dtype=None):
    if dtype is not None:
        x = x.to(dtype)
    x = x.contiguous()
    return LogSoftmax.apply(x, dim)

def calculate_settings_a(n_cols):
    # Heuristic to determine block size and number of warps
    if n_cols < 128:
        BLOCK_SIZE = 32
        num_warps = 2
    elif n_cols < 512:
        BLOCK_SIZE = 128
        num_warps = 4
    else:
        BLOCK_SIZE = 256
        num_warps = 8
    return BLOCK_SIZE, num_warps
