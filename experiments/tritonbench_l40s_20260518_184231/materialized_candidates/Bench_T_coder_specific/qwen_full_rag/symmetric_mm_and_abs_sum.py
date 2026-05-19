import torch
import triton
import triton.language as tl


@triton.jit
def _symmetric_mm_and_abs_sum_kernel(
    A,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offset_m = block_start + tl.arange(0, BLOCK_SIZE)
    offs_M = offset_m[:, None]
    offs_N = tl.arange(0, BLOCK_SIZE)[None, :]
    mask = offs_M < M

    # Compute indices for A
    aset_0 = offs_M % N
    aset_1 = (offs_M // N) + offs_N
    aset = aset_0 * stride_am + aset_1 * stride_ak

    # Load data
    a = tl.load(A + aset, mask=mask, other=0.)

    # Compute y = alpha * A * x + beta * C
    c = tl.dot(a, tl.trans(a))

    # Scale and apply dropout
    c *= alpha

    # Load and scale C
    c += beta * tl.load(C + offs_M, mask=offset_m < M)

    # Write-back output
    tl.store(C + offs_M, c, mask=offset_m < M)


def _symmetric_mm_and_abs_sum_triton(
    A: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    beta: float
) -> torch.Tensor:
    """
    Wrapper function for symmetric matrix multiplication and absolute sum calculation using Triton
    :param A (torch.Tensor): Input matrix of shape (n, m)
    :param C (torch.Tensor): Output matrix of the same shape as alpha * torch.mm(A, A.t())
    :param alpha (float): Scaling factor for the matrix product
    :param beta (float): Scaling factor for matrix C
    :return (torch.Tensor): Scalar tensor containing the sum of absolute values of the resulting matrix C
    """

    # Prepare constants
    M, N = A.shape

    # Make sure A is contiguous if needed
    if not A.is_contiguous():
        A = A.contiguous()

    # Calculate the absolute sum
    asum = torch.zeros(1, dtype=torch.float32, device=A.device)

    # Configuration
    config = _get_config(M, N)

    # Call Triton kernel
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )

    _symmetric_mm_and_abs_sum_kernel[grid](
        A,
        C,
        M,
        N,
        **config
    )

    # Sum absolute values across all blocks
    asum += torch.sum(torch.abs(C))

    return asum
