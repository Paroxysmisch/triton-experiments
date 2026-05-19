import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 32}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 32}, num_stages=8, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 32}, num_stages=16, num_warps=16),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 32}, num_stages=32, num_warps=32),
    ],
    key=['N', 'ND'],
)
@triton.jit
def logsumexp_fwd_kernel(
    x_ptr,  # input tensor
    z_ptr,  # output tensor
    scale_ptr,  # scale tensor (if scale is applied)
    N,  # number of elements in the first dimension
    ND,  # number of elements in the second dimension
    BLOCK_SIZE_M: tl.constexpr,  # block size in the first dimension
    BLOCK_SIZE_K: tl.constexpr,  # block size in the second dimension
    BLOCK_SIZE_N: tl.constexpr,  # block size in the third dimension
):
    pid = tl.program_id(axis=0)
    i_n = pid // ND
    i_d = pid % ND
    o_d = i_n * ND + i_d
    m_d = i_d * BLOCK_SIZE_N

    # Load data
    x = tl.load(x_ptr + o_d * BLOCK_SIZE_N + m_d, mask=m_d < BLOCK_SIZE_N, other=-float('inf'))
    if scale_ptr is not None:
        scale = tl.load(scale_ptr + o_d * BLOCK_SIZE_N + m_d, mask=m_d < BLOCK_SIZE_N, other=1.0)
        x = x * scale

    # Compute max value in block
    b_m = tl.max(x, axis=0)
    x = x - b_m

    # Compute sum of exp(x)
    exp_sum = tl.sum(tl.exp(x), axis=0)

    # Store result
    z = tl.log(exp_sum) + b_m
    tl.store(z_ptr + o_d * BLOCK_SIZE_N + m_d, z, mask=m_d < BLOCK_SIZE_N)

@triton.jit
def logsumexp_fwd(
    x,  # input tensor
    scale=None,  # scale tensor (if scale is applied)
    dtype_out=None,  # desired output data type
):
    N, D = x.shape
    B = 32  # block size
    ND = D // B

    # Reshape input tensor
    x = x.reshape(N, ND, B)

    # Allocate output tensor
    z = tl.zeros((N, ND), dtype=x.dtype)

    # Launch kernel
    grid = (N * ND,)
    logsumexp_fwd_kernel[grid](x, z, scale, N, ND)

    # Reduce along the last dimension
    z = tl.reduce(tl.sum, z, axis=1, keepdim=True)

    # Cast to desired output data type if specified
    if dtype_out is not None:
        z = z.to(dtype_out)

    return z
