import torch
import triton
import triton.language as tl

@triton.jit
def _symmetric_matrix_vector_norm_kernel(
        A_ptr, x_ptr, y_ptr,
        A_stride, x_stride, y_stride,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    A_ptr += row_idx * A_stride
    x_ptr += row_idx * x_stride
    y_ptr += row_idx * y_stride

    x = tl.load(x_ptr + col_offsets, mask=mask).to(tl.float32)
    A = tl.load(A_ptr + col_offsets, mask=mask).to(tl.float32)

    y = A * x
    tl.store(y_ptr + col_offsets, y, mask=mask)


def _symmetric_matrix_vector_norm_triton(
        A: torch.Tensor,
        x: torch.Tensor,
        alpha: float,
        beta: float,
        p: float = 2.0
) -> torch.Tensor:
    r"""
    Wrapper function for smelu triton kernel
    Compute y = alpha * torch.mv(A, x) + beta * y, assume A is symmetric, then calculate \|y\>_p
    :param A (torch.Tensor): Symmetric matrix of shape `(n, n)`
    :param x (torch.Tensor): Vector of shape `(n,)`
    :param alpha (float): Scalar multiplier for the matrix-vector product
    :param beta (float): Scalar multiplier added to `y`
    :param p (float, optional): Order of the norm. Default is 2.0 (Euclidean norm)
    :return (torch.Tensor): The norm of vector `y`
    """

    assert A.is_contiguous()
    assert x.is_contiguous()

    A_dim, x_dim = A.shape[-1], x.shape[-1]
    assert A_dim == x_dim

    y = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(A_dim)
    grid = (A_dim,)
    num_warps = 1

    _symmetric_matrix_vector_norm_kernel[grid](
        A, x, y,
        A.stride(0), x.stride(0), y.stride(0),
        N=A_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    return torch.norm(y, p=p)
