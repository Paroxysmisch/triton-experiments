import triton
import triton.language as tl
import torch

# Triton kernel
@triton.jit
def _quantize_global_transpose(
    A_ptr, B_ptr, absmax_inv_ptr,
    stride_am, stride_an, stride_bm, stride_bn,
    M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr
):
    # Program ID and group partitioning
    pid = tl.program_id(0)
    pid_n = pid % (N // BLOCK_N)
    pid_m = (pid // (N // BLOCK_N)) % GROUP_M
    pid_m = pid_m + (pid // (GROUP_M * (N // BLOCK_N))) * GROUP_M

    # Compute the starting row and column for this block
    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N

    # Check bounds
    is_valid_m = block_start_m + tl.arange(0, BLOCK_M) < M
    is_valid_n = block_start_n + tl.arange(0, BLOCK_N) < N

    # Load input block from A
    A = tl.load(
        A_ptr + block_start_m * stride_am + block_start_n * stride_an,
        mask=is_valid_m[:, None] & is_valid_n[None, :],
        other=0.0
    )

    # Load absmax_inv
    absmax_inv = tl.load(absmax_inv_ptr)

    # Quantize to int8 range
    A_quantized = (A * absmax_inv).to(tl.int8)

    # Transpose and store in B
    tl.store(
        B_ptr + block_start_n * stride_bm + block_start_m * stride_bn,
        A_quantized,
        mask=is_valid_m[:, None] & is_valid_n[None, :]
    )

# Wrapper function
def quantize_global_transpose(A, BLOCK_M=128, BLOCK_N=128, GROUP_M=4):
    """
    Perform global quantization and transposition on matrix A.
    
    Args:
        A (torch.Tensor): Input matrix (2D tensor) of shape [M, N].
        BLOCK_M (int): Block size along M dimension.
        BLOCK_N (int): Block size along N dimension.
        GROUP_M (int): Number of groups for partitioning along M dimension.
        
    Returns:
        torch.Tensor: Quantized and transposed matrix.
    """
    # Ensure A is a 2D tensor
    assert A.ndim == 2, "Input matrix A must be 2D"
    M, N = A.shape

    # Compute absmax and its reciprocal
    absmax = A.abs().max().item()
    absmax_inv = 1.0 / absmax

    # Allocate output tensor B
    B = torch.empty((N, M), dtype=torch.int8, device=A.device)

    # Launch kernel
    grid = lambda META: (
        (M + META['BLOCK_M'] - 1) // META['BLOCK_M'] * 
        (N + META['BLOCK_N'] - 1) // META['BLOCK_N']
    )
    _quantize_global_transpose[grid](
        A_ptr=A,
        B_ptr=B,
        absmax_inv_ptr=torch.tensor([absmax_inv], device=A.device),
        stride_am=A.stride(0), stride_an=A.stride(1),
        stride_bm=B.stride(0), stride_bn=B.stride(1),
        M=M, N=N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, GROUP_M=GROUP_M
    )
    return B
