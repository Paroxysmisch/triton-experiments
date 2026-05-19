import triton
import triton.language as tl
import torch

@triton.jit
def _swiglu_fwd_kernel(X, Y, OUT, M, N, stride_xm, stride_xn, stride_ym, stride_yn, stride_outm, stride_outn, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row = pid

    # Define the column start index for this block
    col_start = tl.arange(0, BLOCK_SIZE)

    # Calculate the index for this block
    x_idx = row * stride_xm + col_start * stride_xn
    y_idx = row * stride_ym + col_start * stride_yn
    out_idx = row * stride_outm + col_start * stride_outn

    # Load data from X and Y
    x = tl.load(X + x_idx, mask=col_start < N)
    y = tl.load(Y + y_idx, mask=col_start < N)

    # Compute sigmoid activation
    sigmoid_x = 1 / (1 + tl.exp(-x))

    # Perform Swiglu operation
    result = x * sigmoid_x * y

    # Store the result
    tl.store(OUT + out_idx, result, mask=col_start < N)

def _swiglu_fwd(xy, M, N):
    # Split input tensor `xy` into `x` and `y`
    x, y = xy[:, :N], xy[:, N:]

    # Ensure the tensors are contiguous
    x = x.contiguous()
    y = y.contiguous()

    # Initialize the output tensor
    out = torch.empty_like(x)

    # Define block size
    BLOCK_SIZE = 128  # This can be tuned

    # Define grid size
    grid = (M,)

    # Launch the kernel
    _swiglu_fwd_kernel[grid](
        x, y, out,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
