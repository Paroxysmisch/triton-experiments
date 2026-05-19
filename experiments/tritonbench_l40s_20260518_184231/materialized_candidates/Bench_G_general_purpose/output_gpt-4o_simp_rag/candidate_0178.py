import torch
import triton
import triton.language as tl

# --- TRITON LOG SOFTMAX KERNEL ---
@triton.jit
def log_softmax_kernel(input_pointer,
                       out_pointer,
                       input_row_stride,
                       out_row_stride,
                       n_cols,
                       BLOCK_SIZE: tl.constexpr):
    # Parallelize across rows
    row_idx = tl.program_id(0)
    row_start_ptr = input_pointer + row_idx * input_row_stride

    # Each thread handles a different element of the row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    # Load the row and normalize by subtracting the max value
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)

    # Compute softmax and then take log
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    log_softmax_output = row_minus_max - tl.log(denominator)

    # Write results to memory
    output_row_start_ptr = out_pointer + row_idx * out_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, log_softmax_output, mask=col_offsets < n_cols)

def calculate_settings(n_cols):
    # Simple heuristic for block size and warps
    BLOCK_SIZE = min(128, n_cols)
    num_warps = min(4, max(1, BLOCK_SIZE // 32))
    return BLOCK_SIZE, num_warps

class LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        n_rows, n_cols = x.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)

        # Allocate output
        y = torch.empty_like(x)

        # Launch the kernel
        log_softmax_kernel[(n_rows,)](
            input_pointer=x,
            out_pointer=y,
            input_row_stride=x.stride(0),
            out_row_stride=y.stride(0),
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )

        # Save for backward
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        (y,) = ctx.saved_tensors

        # Calculate gradient
        grad_input = grad_output - grad_output.sum(dim=1, keepdim=True) * torch.exp(y)
        return grad_input

def log_softmax(x):
    # Ensure input is contiguous
    if not x.is_contiguous():
        x = x.contiguous()

    # Apply LogSoftmax
    return LogSoftmax.apply(x)
