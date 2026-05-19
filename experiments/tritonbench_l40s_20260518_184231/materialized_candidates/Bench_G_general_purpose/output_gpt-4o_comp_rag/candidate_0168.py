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
    row_minus_max = row - tl.max(row, axis=0)  # Normalize by the max to avoid overflow

    # Actual log softmax calculation
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    log_softmax_output = row_minus_max - tl.log(denominator)

    # Write the results to memory
    output_row_start_ptr = out_pointer + row_idx * out_row_stride  # Calculate the start pointer of the current row in the output data.
    output_ptrs = output_row_start_ptr + col_offsets  # Calculate pointers for each element in the output row
    tl.store(output_ptrs, log_softmax_output, mask=col_offsets < n_cols)  # Store the log softmax results in the appropriate locations in GPU memory


@triton.jit
def log_softmax_backward_kernel(grad_output_pointer,
                                output_pointer,
                                grad_input_pointer,
                                input_row_stride,
                                output_row_stride,
                                grad_input_row_stride,
                                n_cols,
                                BLOCK_SIZE: tl.constexpr,
                                ):
    # the rows of the log softmax are independent
    # so we parallelize across those
    row_idx = tl.program_id(0)

    # stride is how much we need to increase the pointer to advance 1 row
    grad_output_row_start_ptr = grad_output_pointer + (row_idx * input_row_stride)
    output_row_start_ptr = output_pointer + (row_idx * output_row_stride)
    grad_input_row_start_ptr = grad_input_pointer + (row_idx * grad_input_row_stride)

    # Each thread within the block will handle a different element of the row.
    col_offsets = tl.arange(0, BLOCK_SIZE)
    grad_output_ptrs = grad_output_row_start_ptr + col_offsets  # Calculate pointers for each element in the current row.
    output_ptrs = output_row_start_ptr + col_offsets  # Calculate pointers for each element in the current row.
    grad_input_ptrs = grad_input_row_start_ptr + col_offsets  # Calculate pointers for each element in the current row.

    # Load the current row data from memory
    grad_output_row = tl.load(grad_output_ptrs, mask=col_offsets < n_cols, other=0.0)
    output_row = tl.load(output_ptrs, mask=col_offsets < n_cols, other=0.0)

    # Actual log softmax backward calculation
    sum_grad_output = tl.sum(grad_output_row, axis=0)
    grad_input_row = grad_output_row - tl.exp(output_row) * sum_grad_output

    # Write the results to memory
    tl.store(grad_input_ptrs, grad_input_row, mask=col_offsets < n_cols)  # Store the gradient results in the appropriate locations in GPU memory


class LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        # Ensure the input is contiguous
        x = x.contiguous()
        M, N = x.shape
        BLOCK_SIZE, num_warps = calculate_settings_b(N)

        # Allocate output
        y = torch.empty_like(x)

        # Launch the forward kernel
        log_softmax_kernel[(M,)](
            input_pointer=x,
            out_pointer=y,
            input_row_stride=x.stride(0),
            out_row_stride=y.stride(0),
            n_cols=N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )

        ctx.save_for_backward(y)
        ctx.dim = dim
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        M, N = grad_output.shape
        BLOCK_SIZE, num_warps = calculate_settings_b(N)

        # Allocate gradient input
        grad_input = torch.empty_like(grad_output)

        # Launch the backward kernel
        log_softmax_backward_kernel[(M,)](
            grad_output_pointer=grad_output,
            output_pointer=y,
            grad_input_pointer=grad_input,
            input_row_stride=grad_output.stride(0),
            output_row_stride=y.stride(0),
            grad_input_row_stride=grad_input.stride(0),
            n_cols=N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )

        return grad_input, None


def log_softmax(x, dim):
    return LogSoftmax.apply(x, dim)


# Helper function to calculate settings based on input size
def calculate_settings_b(n_cols):
    BLOCK_SIZE = 128
    num_warps = 4
    if n_cols > 128:
        BLOCK_SIZE = 256
        num_warps = 8
    return BLOCK_SIZE, num_warps
