import torch
import triton
import triton.language as tl

@triton.jit
def pseudoinverse_kernel(
    A_ptr, U_ptr, S_ptr, V_ptr, A_pinv_ptr,
    m, n, batch_stride, full_matrices, rcond, BLOCK_SIZE: tl.constexpr
):
    # Compute the thread indices
    batch_idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    col_idx = tl.program_id(2)

    # Load matrix A from global memory
    A = tl.load(A_ptr + batch_idx * batch_stride + row_idx * n + col_idx)

    # Perform SVD on matrix A
    U, S, V = tl.svd(A, full_matrices=full_matrices)

    # Invert the singular values with the given rcond
    max_singular_value = tl.max(S)
    S_inv = tl.where(S > rcond * max_singular_value, 1.0 / S, 0.0)

    # Compute the pseudoinverse: V^H * S^+ * U^H
    A_pinv = tl.matmul(V, tl.diag(S_inv))
    A_pinv = tl.matmul(A_pinv, tl.transpose(U))

    # Store the result in global memory
    tl.store(A_pinv_ptr + batch_idx * batch_stride + row_idx * n + col_idx, A_pinv)

def pseudoinverse_svd(A, *, full_matrices=True, rcond=1e-15, out=None):
    """
    Computes the Moore-Penrose pseudoinverse of a matrix using SVD.

    Args:
        A (Tensor): Input tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.

    Keyword args:
        full_matrices (bool, optional): If `True` (default), compute the full SVD. If `False`, compute the reduced SVD.
        rcond (float, optional): Relative condition number threshold. Singular values smaller than `rcond * largest_singular_value` are set to zero. Default: `1e-15`.
        out (Tensor, optional): Output tensor. Ignored if `None`. Default: `None`.

    Returns:
        Tensor: The pseudoinverse of the input matrix.
    """
    # Ensure the input is a float tensor
    if not A.is_floating_point():
        raise TypeError("Input must be a floating point tensor")

    # Get the shape of the input matrix
    *batch_dims, m, n = A.shape
    batch_size = torch.prod(torch.tensor(batch_dims)).item()

    # Prepare the output tensor
    if out is None:
        out = torch.empty((*batch_dims, n, m), dtype=A.dtype, device=A.device)

    # Launch the Triton kernel
    grid = (batch_size, m, n)
    pseudoinverse_kernel[grid](
        A_ptr=A.data_ptr(),
        U_ptr=None,  # Placeholder, SVD will handle this internally
        S_ptr=None,  # Placeholder, SVD will handle this internally
        V_ptr=None,  # Placeholder, SVD will handle this internally
        A_pinv_ptr=out.data_ptr(),
        m=m, n=n,
        batch_stride=m * n,
        full_matrices=full_matrices,
        rcond=rcond,
        BLOCK_SIZE=16  # Example block size
    )

    return out
