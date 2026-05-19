import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Define the program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate block indices
    block_m = pid_m * BLOCK_SIZE_M
    block_n = pid_n * BLOCK_SIZE_N

    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks of A and B
        a_block = tl.load(a_ptr + (block_m + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_am + (k + tl.arange(0, BLOCK_SIZE_K)) * stride_ak, mask=block_m + tl.arange(0, BLOCK_SIZE_M) < M)
        b_block = tl.load(b_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * stride_bk + (block_n + tl.arange(0, BLOCK_SIZE_N)) * stride_bn, mask=block_n + tl.arange(0, BLOCK_SIZE_N) < N)

        # Perform matrix multiplication for the block
        acc += tl.dot(a_block, b_block)

    # Store the result
    c_block = acc.to(tl.float32)
    tl.store(c_ptr + (block_m + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_cm + (block_n + tl.arange(0, BLOCK_SIZE_N)) * stride_cn, c_block, mask=block_m + tl.arange(0, BLOCK_SIZE_M) < M)

def matmul(a, b, c, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32):
    # Get the shapes and strides
    M, K = a.shape
    K, N = b.shape
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    # Launch the kernel
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    matmul_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        M=M, N=N, K=K,
        stride_am=stride_am, stride_ak=stride_ak,
        stride_bk=stride_bk, stride_bn=stride_bn,
        stride_cm=stride_cm, stride_cn=stride_cn,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )

# Example usage
a = torch.randn(256, 256, device='cuda', dtype=torch.float32)
b = torch.randn(256, 256, device='cuda', dtype=torch.float32)
c = torch.empty(256, 256, device='cuda', dtype=torch.float32)
matmul(a, b, c)
