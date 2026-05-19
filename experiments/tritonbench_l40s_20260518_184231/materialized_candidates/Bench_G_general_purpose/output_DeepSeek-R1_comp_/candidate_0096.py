import torch
import triton
import triton.language as tl

@triton.jit
def _rms_norm_fwd_fused(
    X,  # Pointer to the input tensor
    Y,  # Pointer to the output tensor
    W,  # Pointer to the weight tensor
    stride_x_row,  # Stride between rows of X
    stride_y_row,  # Stride between rows of Y
    stride_w,      # Stride of the weight tensor (should be 1 for contiguous)
    N,             # Number of columns in X
    eps,           # Small epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,  # Block size (power of two)
):
    row = tl.program_id(0)  # Row index for the current program instance

    # Compute pointers to the current row in X and Y
    x_row_ptr = X + row * stride_x_row
    y_row_ptr = Y + row * stride_y_row

    # Compute sum of squares for the current row
    sum_sq = 0.0
    for col_offset in range(0, N, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x)

    # Compute variance and reciprocal standard deviation (rstd)
    var = sum_sq / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize the row and multiply by weights
    for col_offset in range(0, N, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0)
        w = tl.load(W + cols * stride_w, mask=mask, other=0.0)
        y = x * rstd * w
        tl.store(y_row_ptr + cols, y, mask=mask)

def rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Ensure the weight tensor is contiguous and has the correct shape
    assert x.shape[-1] == weight.shape[-1], "Dimension mismatch between x and weight"
    weight = weight.contiguous()

    # Flatten the input tensor to 2D (M, N) where M is the product of all but the last dimension
    x_2d = x.reshape(-1, x.size(-1))
    M, N = x_2d.shape

    # Allocate output tensor
    y = torch.empty_like(x)
    y_2d = y.view(-1, N)

    # Determine the block size, ensuring it's a power of two and <= 1024
    BLOCK_SIZE = triton.next_power_of_2(N)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024

    # Heuristic for number of warps based on block size
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    elif BLOCK_SIZE >= 1024:
        num_warps = 8
    elif BLOCK_SIZE >= 512:
        num_warps = 4
    else:
        num_warps = 2

    # Launch the Triton kernel
    _rms_norm_fwd_fused[(M,)](x_2d, y_2d, weight,
                              x_2d.stride(0), y_2d.stride(0), weight.stride(0),
                              N, eps,
                              BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)

    return y
