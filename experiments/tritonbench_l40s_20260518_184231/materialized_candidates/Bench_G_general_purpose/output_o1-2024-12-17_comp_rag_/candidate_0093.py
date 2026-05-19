import triton
import triton.language as tl
import torch

# Optional utility to multiply two tensors element-wise. You can inline this if preferred.
class kernel_utils:
    @staticmethod
    @triton.jit
    def mul(x, y):
        return x * y

@triton.jit
def matmul_kernel(
    C, A, B,
    M, N, K,
    stride_cm, stride_cn,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for sub-matrices
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k  = tl.arange(0, BLOCK_K)

    # Create pointers for A and B blocks
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator to zero
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Bound check for partial blocks along K
        k_bound = K - k * BLOCK_K
        a_block = tl.load(a_ptrs, mask=offs_k[None, :] < k_bound, other=0.)
        b_block = tl.load(b_ptrs, mask=offs_k[:, None] < k_bound, other=0.)

        # Compute partial product
        accumulator += tl.dot(a_block, b_block)

        # Advance to next K-block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Elementwise mul to replicate C = (A x B) * (A x B)
    c_val = kernel_utils.mul(accumulator, accumulator)
    c_val = tl.cast(c_val, tl.float16)

    # Compute pointers for storing into C
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs  = C + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)

    # Store the result
    mask_m = offs_cm < M
    mask_n = offs_cn < N
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, c_val, mask=out_mask)

def matmul(a: torch.Tensor, b: torch.Tensor, block_m=128, block_n=128, block_k=32):
    """
    Matrix multiplication wrapper that computes:
       C = (A x B) * (A x B)
    using blocked Triton kernel.
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matrix multiplication."

    M, K = a.shape
    K_, N = b.shape
    assert K == K_, "Matrix dimensions must match."

    # Allocate output
    c = torch.zeros((M, N), device=a.device, dtype=torch.float16)

    # Extract strides (assuming row-major / contiguous for demonstration)
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    # Determine grid size
    grid_m = (M + block_m - 1) // block_m
    grid_n = (N + block_n - 1) // block_n

    # Launch Triton kernel
    matmul_kernel[grid_m, grid_n](
        c, a, b,
        M, N, K,
        stride_cm, stride_cn,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k
    )
    return c
