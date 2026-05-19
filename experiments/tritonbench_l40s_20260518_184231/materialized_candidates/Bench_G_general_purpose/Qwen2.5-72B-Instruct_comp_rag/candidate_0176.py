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
    row_max = tl.max(row, axis=0)  # Find the maximum value in the row to stabilize the computation
    row_minus_max = row - row_max  # Normalize by the max to avoid overflow

    # Compute the exponentials and their sum
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    log_denominator = tl.log(denominator)

    # Compute the log softmax output
    log_softmax_output = row_minus_max - log_denominator

    # Write the results to memory
    output_row_start_ptr = out_pointer + row_idx * out_row_stride  # Calculate the start pointer of the current row in the output data.
    output_ptrs = output_row_start_ptr + col_offsets  # Calculate pointers for each element in the output row
    tl.store(output_ptrs, log_softmax_output, mask=col_offsets < n_cols)  # Store the log softmax results in the appropriate locations in GPU memory

@triton.jit
def log_softmax_backward_kernel(grad_output_pointer,
                                output_pointer,
                                grad_input_pointer,
                                grad_output_row_stride,
                                output_row_stride,
                                grad_input_row_stride,
                                n_cols,
                                BLOCK_SIZE: tl.constexpr,
                                ):
    # the rows of the log softmax are independent
    # so we parallelize across those
    row_idx = tl.program_id(0)

    # stride is how much we need to increase the pointer to advance 1 row
    grad_output_row_start_ptr = grad_output_pointer + (row_idx * grad_output_row_stride)
    output_row_start_ptr = output_pointer + (row_idx * output_row_stride)

    # Each thread within the block will handle a different element of the row.
    col_offsets = tl.arange(0, BLOCK_SIZE)
    grad_output_ptrs = grad_output_row_start_ptr + col_offsets  # Calculate pointers for each element in the current row.
    output_ptrs = output_row_start_ptr + col_offsets  # Calculate pointers for each element in the current row.

    # Load the current row data from memory
    grad_output_row = tl.load(grad_output_ptrs, mask=col_offsets < n_cols, other=0.0)
    output_row = tl.load(output_ptrs, mask=col_offsets < n_cols, other=-float('inf'))

    # Compute the gradient of the input
    grad_input_row = grad_output_row - tl.sum(grad_output_row * tl.exp(output_row), axis=0)

    # Write the results to memory
    grad_input_row_start_ptr = grad_input_pointer + row_idx * grad_input_row_stride  # Calculate the start pointer of the current row in the output data.
    grad_input_ptrs = grad_input_row_start_ptr + col_offsets  # Calculate pointers for each element in the output row
    tl.store(grad_input_ptrs, grad_input_row, mask=col_offsets < n_cols)  # Store the log softmax results in the appropriate locations in GPU memory

class LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        ctx.dim = dim
        ctx.save_for_backward(input)

        n_rows, n_cols = input.shape
        BLOCK_SIZE, num_warps = calculate_settings_a(n_cols)

        # Allocate output
        output = torch.empty_like(input)

        # Launch the kernel with calculated settings
        log_softmax_kernel[(n_rows,)](
            input_pointer=input,
            out_pointer=output,
            input_row_stride=input.stride(0),
            out_row_stride=output.stride(0),
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        dim = ctx.dim

        n_rows, n_cols = input.shape
        BLOCK_SIZE, num_warps = calculate_settings_a(n_cols)

        # Allocate output
        grad_input = torch.empty_like(input)

        # Launch the kernel with calculated settings
        log_softmax_backward_kernel[(n_rows,)](
            grad_output_pointer=grad_output,
            output_pointer=input,
            grad_input_pointer=grad_input,
            grad_output_row_stride=grad_output.stride(0),
            output_row_stride=input.stride(0),
            grad_input_row_stride=grad_input.stride(0),
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )
        return grad_input, None

def log_softmax(x, dim, dtype=None):
    if dtype is not None:
        x = x.to(dtype)
    if not x.is_contiguous():
        x = x.contiguous()
    return LogSoftmax.apply(x, dim)

# Helper function to calculate block size and number of warps
def calculate_settings_a(n_cols):
    if n_cols < 1024:
        return 128, 4
    else:
        return 256, 8

# Example usage
if __name__ == "__main__":
    x = torch.randn(10, 1000, device='cuda')
    dim = 1
    output = log_softmax(x, dim)
    print(output)
