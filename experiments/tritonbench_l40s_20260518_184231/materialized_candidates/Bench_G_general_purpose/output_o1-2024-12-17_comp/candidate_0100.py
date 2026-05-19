import torch
import triton
import triton.language as tl


@triton.jit
def leaky_relu(x, negative_slope):
    return tl.where(x >= 0, x, x * negative_slope)


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32},
            num_stages=2,
            num_warps=4
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32},
            num_stages=2,
            num_warps=8
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32},
            num_stages=2,
            num_warps=8
        ),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K, 
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    activation,
    negative_slope,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the row and col of C this program will produce
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Create pointers for A and B
    a_ptrs = a_ptr + (rm[:, None] * stride_am + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_ak)
    b_ptrs = b_ptr + (tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Block level matrix multiplication
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (k + tl.arange(0, BLOCK_SIZE_K)[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k + tl.arange(0, BLOCK_SIZE_K)[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Optionally apply activation
    if activation != 0:
        acc = leaky_relu(acc, negative_slope)

    # Write back to C
    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def matmul(a, b, activation=False, negative_slope=0.01):
    # Ensure input is on GPU
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Tensors must be on GPU.")

    # Shapes
    M, K = a.shape
    K_b, N = b.shape

    # Check dimension compatibility
    if K != K_b:
        raise ValueError("Incompatible dimensions for matrix multiplication.")

    # Ensure contiguity
    a = a.contiguous()
    b = b.contiguous()

    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (
        (M + META['BLOCK_SIZE_M'] - 1) // META['BLOCK_SIZE_M'],
        (N + META['BLOCK_SIZE_N'] - 1) // META['BLOCK_SIZE_N']
    )

    matmul_kernel[grid](
        a_ptr=a, b_ptr=b, c_ptr=c,
        M=M, N=N, K=K,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bk=b.stride(0), stride_bn=b.stride(1),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        activation=int(activation),
        negative_slope=negative_slope
    )
    return c
