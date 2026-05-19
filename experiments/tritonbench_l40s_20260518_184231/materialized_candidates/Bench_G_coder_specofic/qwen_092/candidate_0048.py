triton
import triton
import triton.language as tl

# Define the kernel function
@triton.jit
def mean_dim_kernel(
    X,  # input tensor
    Mean,  # output tensor
    M,  # number of rows to process
    N,  # number of columns to process
    BLOCK_M: tl.constexpr,  # block size in the M dimension
    BLOCK_N: tl.constexpr,  # block size in the N dimension
):
    pid = tl.program_id(axis=0)
    row = pid * BLOCK_M
    row_mask = row < M
    x = tl.load(X + row[:, None] * N)
    _mean = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for n in range(BLOCK_N):
        _mean += x[:, n]
    _mean /= N
    tl.store(Mean + row[:, None] * BLOCK_N, _mean, mask=row_mask[:, None])

# Define the dim_compress function
def dim_compress(inp, dims):
    return tl.reshape(inp, tuple(dims) + (-1,))

# Define the mean_dim wrapper function
def mean_dim(x, dim, keepdim=True):
    x = dim_compress(x, dim)
    M = x.shape[0]
    N = x.shape[1]
    out = tl.zeros((M, N), dtype=x.dtype)
    grid = (triton.cdiv(M, BLOCK_M),)
    mean_dim_kernel[grid](x, out, M, N, BLOCK_M, BLOCK_N)
    if not keepdim:
        out = tl.squeeze(out, dim)
    return out

# Define block sizes
BLOCK_M = 32
BLOCK_N = 32
