import triton
import triton.language as tl
import torch

@triton.jit
def _quantize_global_transpose(
    A_ptr, B_ptr, absmax_inv_ptr,
    stride_am, stride_an, stride_bm, stride_bn,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create block-level offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create masks for bounds checking
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Load absmax_inv scalar
    absmax_inv = tl.load(absmax_inv_ptr)

    # Compute input offsets and load data
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    # Quantize the values
    # Scale to int8 range (-127 to 127) and clip
    scaled = tl.clip(a * absmax_inv * 127.0, -127.0, 127.0)
    quantized = tl.cast(scaled, tl.int8)

    # Store transposed result
    # Note: We swap m and n indices for transposition
    b_ptrs = B_ptr + offs_n[:, None] * stride_bm + offs_m[None, :] * stride_bn
    tl.store(b_ptrs, quantized.T, mask=mask_n[:, None] & mask_m[None, :])

def quantize_global_transpose(A: torch.Tensor, absmax_inv: torch.Tensor) -> torch.Tensor:
    """
    Quantize and transpose a matrix using Triton.
    
    Args:
        A: Input tensor (float32)
        absmax_inv: Inverse of maximum absolute value for scaling
    
    Returns:
        B: Quantized and transposed tensor (int8)
    """
    assert A.is_cuda and absmax_inv.is_cuda
    M, N = A.shape
    B = torch.empty((N, M), dtype=torch.int8, device=A.device)

    # Define block sizes
    BLOCK_M = 32
    BLOCK_N = 32

    # Calculate grid dimensions
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)

    # Launch kernel
    _quantize_global_transpose[grid](
        A_ptr=A, 
        B_ptr=B,
        absmax_inv_ptr=absmax_inv,
        stride_am=A.stride(0),
        stride_an=A.stride(1),
        stride_bm=B.stride(0),
        stride_bn=B.stride(1),
        M=M, N=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )

    return B
