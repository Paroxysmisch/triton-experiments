import torch
import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X,  # pointer to the input tensor
    Y,  # pointer to the output tensor
    stride_x_row,  # stride between rows of X
    N,  # number of columns in X
    eps,  # small constant to avoid division by zero
    BLOCK_N: tl.constexpr,  # compile-time constant block size for columns
):
    # Determine the row index for this program
    pid = tl.program_id(0)
    # Compute the starting address of the current row in X and Y
    row_start = pid * stride_x_row
    # Generate offsets for the entire block of columns
    offsets = row_start + tl.arange(0, BLOCK_N)
    # Create mask to avoid out-of-bounds access (columns beyond N are masked)
    mask = tl.arange(0, BLOCK_N) < N
    # Load the current block of data from X
    x = tl.load(X + offsets, mask=mask, other=0.0)
    # Compute sum of squares for the row
    sum_sq = tl.sum(x * x, axis=0)
    # Compute reciprocal of sqrt(variance + eps)
    rstd = 1.0 / tl.sqrt(sum_sq + eps)
    # Normalize the block and store to Y
    y = x * rstd
    tl.store(Y + offsets, y, mask=mask)

def _l2_norm_fwd(x: torch.Tensor, eps: float) -> torch.Tensor:
    # Reshape input to 2D (M, N) and make contiguous
    x_ = x.reshape(-1, x.size(-1))
    M, N = x_.shape
    if not x_.is_contiguous():
        x_ = x_.contiguous()
    # Initialize output tensor
    y = torch.empty_like(x_)
    # Determine the maximum block size based on element size (64KB limit)
    element_size = x_.element_size()
    max_block_elements = (64 * 1024) // element_size
    BLOCK_N = max_block_elements
    # Check if the feature dimension exceeds the block size
    if N > BLOCK_N:
        raise ValueError(f"Feature dimension {N} exceeds maximum block size {BLOCK_N}. Consider reducing the feature dimension or using a smaller data type.")
    # Launch the kernel with M blocks, each processing one row
    grid = (M,)
    _l2_norm_fwd_1pass_kernel[grid](
        x_, y, x_.stride(0), N, eps,
        BLOCK_N=BLOCK_N
    )
    # Reshape output to match the original input shape
    return y.reshape(x.shape)
