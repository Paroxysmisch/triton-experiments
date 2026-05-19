import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N"],
)
@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    stride_x_row,  # row stride of the input
    N,  # number of columns in X
    eps,  # epsilon for numerical stability
    BLOCK_N: tl.constexpr,  # compile-time constant block size
):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_x_row
    cols = tl.arange(0, BLOCK_N)
    # Load the data for the current row, masking out-of-bounds elements
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    x_zeros = tl.where(cols < N, x, 0.0)
    # Compute sum of squares for the row
    var = tl.sum(x_zeros * x_zeros, axis=0)
    # Reciprocal of the standard deviation (with eps for stability)
    rstd = 1.0 / tl.sqrt(var + eps)
    # Normalize and store the result
    y = x * rstd
    tl.store(Y + cols, y, mask=cols < N)

def _l2_norm_fwd(x, eps=1e-6):
    original_shape = x.shape
    x = x.reshape(-1, x.size(-1))  # Reshape to 2D if necessary
    # Ensure contiguous memory layout for the last dimension
    if x.stride(-1) != 1:
        x = x.contiguous()
    M, N = x.shape
    # Initialize output tensor
    y = torch.empty_like(x)
    # Determine the block size, considering element size and 64KB limit
    element_size = x.element_size()
    max_fused_size = 65536 // element_size
    BLOCK_N = min(max_fused_size, triton.next_power_of_2(N))
    # Check if feature dimension is supported
    if N > BLOCK_N:
        raise ValueError(f"Feature dimension {N} exceeds maximum supported size {BLOCK_N}.")
    # Launch the kernel with one block per row
    grid = (M,)
    _l2_norm_fwd_1pass_kernel[grid](x, y, x.stride(0), N, eps, BLOCK_N)
    # Reshape output to match original input shape
    return y.reshape(original_shape)
