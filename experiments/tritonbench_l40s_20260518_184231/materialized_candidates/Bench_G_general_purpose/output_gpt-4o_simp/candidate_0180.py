import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(X_ptr, Y_ptr, stride_x_row, N, eps, BLOCK_N, **meta):
    # Get the row index for this program instance
    row = tl.program_id(0)

    # Compute the start index for this row in the input and output tensors
    offset = row * stride_x_row

    # Initialize accumulators for variance calculation
    sum_sq = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Iterate over the columns in blocks of BLOCK_N
    for start in range(0, N, BLOCK_N):
        # Load a block of data from X
        block_offsets = tl.arange(0, BLOCK_N) + start
        mask = block_offsets < N  # Mask to handle boundaries
        x = tl.load(X_ptr + offset + block_offsets, mask=mask, other=0.0)

        # Accumulate the sum of squares
        sum_sq += x * x

    # Compute the L2 norm
    norm = tl.sqrt(sum_sq + eps)

    # Normalize the input data and store in Y
    for start in range(0, N, BLOCK_N):
        block_offsets = tl.arange(0, BLOCK_N) + start
        mask = block_offsets < N
        x = tl.load(X_ptr + offset + block_offsets, mask=mask, other=0.0)
        y = x / norm
        tl.store(Y_ptr + offset + block_offsets, y, mask=mask)

def _l2_norm_fwd(x, eps=1e-5, BLOCK_N=128):
    # Ensure input is contiguous
    if not x.is_contiguous():
        x = x.contiguous()

    # Get input dimensions
    num_rows, num_cols = x.shape

    # Prepare output tensor
    y = torch.empty_like(x)

    # Launch the Triton kernel
    grid = (num_rows,)  # One program instance per row
    stride_x_row = x.stride(0)

    _l2_norm_fwd_1pass_kernel[grid](
        x, y, stride_x_row, num_cols, eps, BLOCK_N,
        num_warps=4,  # Set an appropriate number of warps
        num_stages=2  # Set an appropriate number of stages
    )

    return y
