import triton
import triton.language as tl

# Triton kernel to perform the Swiglu operation
@triton.jit
def _swiglu_fwd_kernel(
    X, Y, OUT,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % grid_m
    pid_n = pid // grid_m

    row = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row = row[:, None]
    col = col[None, :]

    mask = row < M and col < N
    x = tl.load(X + row * N + col, mask=mask)
    y = tl.load(Y + row * N + col, mask=mask)
    sigmoid_x = 1 / (1 + tl.exp(-x))
    out = x * sigmoid_x * y
    tl.store(OUT + row * N + col, out, mask=mask)

# Wrapper function to prepare input tensors and launch the kernel
def _swiglu_fwd(xy, device):
    M, N = xy.shape
    x, y = xy[:, :N//2], xy[:, N//2:]

    # Ensure tensors are contiguous
    x = x.contiguous()
    y = y.contiguous()

    # Initialize output tensor
    out = torch.zeros_like(x)

    # Define block size
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 128

    # Calculate grid size
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    grid = (grid_m * grid_n,)

    # Launch the kernel
    _swiglu_fwd_kernel[grid](x, y, out, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)

    return out
