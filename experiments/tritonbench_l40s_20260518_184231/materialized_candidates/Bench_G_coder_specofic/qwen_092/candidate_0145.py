import triton
import triton.language as tl

@triton.jit
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 512}, num_stages=2, num_warps=4),
    ],
    key=['K', 'N'],
)
def dequantize_kernel(
    b: tl.tensor,  # input int8 matrix
    b_scale: tl.tensor,  # scale factors
    fpb: tl.tensor,  # output float matrix
    K: tl.constexpr,  # number of columns in b
    N: tl.constexpr,  # number of rows in b and columns in fpb
    stride_b: tl.constexpr,  # stride of b
    stride_b_scale: tl.constexpr,  # stride of b_scale
    stride_fpb: tl.constexpr,  # stride of fpb
):
    pid = tl.program_id(axis=0)
    num_pid = tl.cdiv(N, BLOCK_SIZE_N)
    n = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k = tl.arange(0, BLOCK_SIZE_K)

    # Load b and b_scale
    b_frag = tl.load(b + n[:, None] * stride_b + k[None, :], mask=n < N and k < K)
    b_scale_frag = tl.load(b_scale + k, mask=k < K)

    # Dequantize
    fpb_frag = b_frag * b_scale_frag

    # Store result in fpb
    tl.store(fpb + n[:, None] * stride_fpb + k[None, :], fpb_frag, mask=n < N and k < K)


def matmul_dequantize_int8(a, b, a_scale, b_scale, c, K, N):
    # Ensure matrix dimension compatibility
    assert a.shape[1] == K, "Dimension mismatch: a.shape[1] != K"
    assert b.shape[0] == K, "Dimension mismatch: b.shape[0] != K"
    assert c.shape[0] == a.shape[0], "Dimension mismatch: c.shape[0] != a.shape[0]"
    assert c.shape[1] == b.shape[1], "Dimension mismatch: c.shape[1] != b.shape[1]"

    # Dequantize b
    b_dequantized = torch.zeros_like(b, dtype=torch.float32)
    dequantize_kernel[b.shape[0], K](b, b_scale, b_dequantized, K, N, b.stride(0), b_scale.stride(0), b_dequantized.stride(0))

    # Perform matrix multiplication
    c[:] = torch.mm(a, b_dequantized)
