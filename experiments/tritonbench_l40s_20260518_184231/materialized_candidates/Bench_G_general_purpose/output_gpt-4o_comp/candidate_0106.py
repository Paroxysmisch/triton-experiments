import triton
import triton.language as tl
import torch

@triton.jit
def iv_dependent_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    type: tl.constexpr
):
    # Program ID determines which block of the matrix we are working on
    pid = tl.program_id(axis=0)

    # Calculate block row and column indices
    num_blocks_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    block_row = pid // num_blocks_m
    block_col = pid % num_blocks_m

    # Offsets for A, B, and C matrices
    row_offset = block_row * BLOCK_SIZE_M
    col_offset = block_col * BLOCK_SIZE_N

    # Create accumulator for C block
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension in blocks of BLOCK_SIZE_K
    for k_offset in range(0, K, BLOCK_SIZE_K):
        # Offsets for the current block of A and B
        a_offset = row_offset * stride_am + k_offset * stride_ak
        b_offset = k_offset * stride_bk + col_offset * stride_bn

        # Load blocks of A and B
        a = tl.load(a_ptr + a_offset + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_am + tl.arange(0, BLOCK_SIZE_K)[None, :], mask=(row_offset + tl.arange(0, BLOCK_SIZE_M) < M)[:, None] & (k_offset + tl.arange(0, BLOCK_SIZE_K) < K), other=0.0)
        b = tl.load(b_ptr + b_offset + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_bk + tl.arange(0, BLOCK_SIZE_N)[None, :], mask=(k_offset + tl.arange(0, BLOCK_SIZE_K) < K)[:, None] & (col_offset + tl.arange(0, BLOCK_SIZE_N) < N), other=0.0)

        # Compute the partial product
        acc += tl.dot(a, b)

    # Write back the result to C
    c_offset = row_offset * stride_cm + col_offset * stride_cn
    c = acc.to(tl.float32)
    tl.store(c_ptr + c_offset + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_cm + tl.arange(0, BLOCK_SIZE_N)[None, :], c, mask=(row_offset + tl.arange(0, BLOCK_SIZE_M) < M)[:, None] & (col_offset + tl.arange(0, BLOCK_SIZE_N) < N))


def iv_dependent_matmul_wrapper(a, b, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, type="default"):
    """
    Wrapper function to configure and launch the Triton kernel.
    """
    # Ensure inputs are on the same device
    assert a.device == b.device, "Input matrices must be on the same device"

    # Allocate output matrix C
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # Define grid size
    grid = lambda META: (
        (M + META['BLOCK_SIZE_M'] - 1) // META['BLOCK_SIZE_M'] *
        (N + META['BLOCK_SIZE_N'] - 1) // META['BLOCK_SIZE_N'],
    )

    # Launch the kernel
    iv_dependent_matmul_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        M=M, N=N, K=K,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bk=b.stride(0), stride_bn=b.stride(1),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        type=type
    )

    return c


# Example Usage
if __name__ == "__main__":
    # Matrix dimensions
    M, N, K = 128, 128, 128

    # Block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    # Create random input matrices
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)

    # Call the wrapper
    c = iv_dependent_matmul_wrapper(a, b, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

    # Validate result
    torch_result = torch.matmul(a, b)
    print("Max difference:", (c - torch_result).abs().max())
