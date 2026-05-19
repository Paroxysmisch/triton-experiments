import triton
import triton.language as tl
import torch


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    ACTIVATION: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Program IDs for block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Ranges for output sub-block
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Create accumulator for partial results
    accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Initial pointers to A and B
    # Iterate over K dimension in steps of BLOCK_SIZE_K
    k_block_offsets = tl.arange(0, BLOCK_SIZE_K)
    A_ptrs = A_ptr + (rm[:, None] * stride_am + k_block_offsets[None, :] * stride_ak)
    B_ptrs = B_ptr + (k_block_offsets[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Accumulate over the K dimension
    for k_offset in range(0, K, BLOCK_SIZE_K):
        # Load A and B tiles
        maskA = (rm[:, None] < M) & ((k_offset + k_block_offsets[None, :]) < K)
        maskB = ((k_offset + k_block_offsets[:, None]) < K) & (rn[None, :] < N)
        a = tl.load(A_ptrs, mask=maskA, other=0.0)
        b = tl.load(B_ptrs, mask=maskB, other=0.0)

        # Perform partial matmul
        accum += tl.dot(a, b)

        # Advance pointers by BLOCK_SIZE_K
        A_ptrs += BLOCK_SIZE_K * stride_ak
        B_ptrs += BLOCK_SIZE_K * stride_bk

    # Optional activation
    if ACTIVATION == 1:  # leaky_relu
        accum = tl.where(accum >= 0, accum, 0.01 * accum)

    # Store results into C
    C_ptrs = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    maskC = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C_ptrs, accum, mask=maskC)


def matmul(A, B, activation='none',
           BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32):
    """
    A: torch.Tensor (M, K)
    B: torch.Tensor (K, N)
    activation: 'none' or 'leaky_relu'
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: Triton block sizes
    """
    # Extract dimensions
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Incompatible dimensions for matrix multiplication"

    # Allocate output
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Compute strides assuming PyTorch row-major
    # If Tensors are not contiguous in row-major, one can pass in manual strides:
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Activation ID (0=none, 1=leaky_relu)
    act_id = 0 if activation == 'none' else 1

    # Grid configuration
    grid = (
       ( (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M ),
       ( (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N )
    )

    # Launch Triton kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        act_id,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    return C
