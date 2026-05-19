import torch
import triton
import triton.language as tl

@triton.jit
def _diagonal_product_kernel(
    matrix_ptr,  # pointer to the matrix
    out_ptr,     # pointer to a single-element output
    stride,      # row stride of the matrix
    n,           # dimension (n x n)
    BLOCK_SIZE: tl.constexpr
):
    """
    Parallel product of the diagonal entries of an n x n matrix.
    matrix[i, i] is located at matrix_ptr + i*stride + i.
    The single accumulated product is written to out_ptr[0].
    This kernel uses one 1D block, so program_id(0) should be 0.
    """
    block_start = 0
    # Thread-local accumulator
    acc = tl.float32(1.0)

    # We iterate in chunks of BLOCK_SIZE to handle large n
    # Each chunk loads diagonal elements in range [block_start, block_start+BLOCK_SIZE)
    # until we cover all n diagonal entries.
    while block_start < n:
        # Load diagonal indices for this chunk
        offs = tl.arange(0, BLOCK_SIZE) + block_start
        mask = offs < n
        # For row-major: element (i, i) = matrix_ptr + i*stride + i
        diag_elem = tl.load(matrix_ptr + offs * stride + offs, mask=mask, other=1.0)
        # Multiply partial product
        for step in range(BLOCK_SIZE // 2):
            half_step = BLOCK_SIZE >> (step + 1)
            diag_elem = diag_elem * tl.pad(diag_elem, [0, half_step])[half_step:]
        # The first element of diag_elem now holds the partial product for this chunk
        acc = acc * diag_elem[0]
        block_start += BLOCK_SIZE

    # Write result to out_ptr
    tl.store(out_ptr, acc)


def _triton_diagonal_product(matrix: torch.Tensor) -> torch.Tensor:
    """
    Computes product of the diagonal elements of an n x n matrix using Triton.
    Returns a scalar tensor containing the product of diagonal entries.
    """
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1], \
        "Input must be a 2D square matrix"
    n = matrix.shape[0]

    # We launch a single block for the diagonal product.
    # We only need one output element to store the final product.
    out = torch.empty((), dtype=matrix.dtype, device=matrix.device)
    # Stride for row-major in PyTorch is (n, 1) for an n x n matrix
    stride = matrix.stride(0)

    BLOCK_SIZE = triton.next_power_of_2(n) if n > 1 else 1
    grid = (1,)

    _diagonal_product_kernel[grid](
        matrix,                      # matrix_ptr
        out,                         # out_ptr
        stride,                      # stride
        n,                           # dimension
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def determinant_via_qr(A, *, mode='reduced', out=None) -> torch.Tensor:
    """
    Computes the determinant of a square matrix A using QR decomposition.
    
    QR Decomposition: A = Q R
    For real A, det(Q) = ±1.
    For complex A, |det(Q)| = 1 (the phase can be non-zero).
    Final: det(A) = det(Q) * (product of diag(R)).
    
    Args:
        A (torch.Tensor): Square matrix of shape (n, n).
        mode (str, optional): 'reduced' or 'complete' for QR. Default 'reduced'.
        out (torch.Tensor, optional): If provided, the determinant is written into this tensor.
    
    Returns:
        (torch.Tensor): Scalar tensor containing det(A).
    """
    # 1) Compute the QR decomposition
    Q, R = torch.linalg.qr(A, mode=mode)

    # 2) Compute determinant of Q. 
    #    For real matrices, this is ±1. For complex, magnitude should be 1, with possible phase.
    det_Q = torch.linalg.det(Q)

    # 3) Compute product of diagonal elements of R via Triton kernel
    prod_diag_R = _triton_diagonal_product(R)

    # 4) Combine for final determinant
    det_val = det_Q * prod_diag_R

    if out is not None:
        out[...] = det_val
        return out
    else:
        return det_val.clone()
