import torch
import triton
import triton.language as tl

@triton.jit
def _matrix_multiply_and_row_dot_kernel(
        A_ptr,
        B_ptr,
        C_ptr,
        alpha: float,
        beta: float,
        n: tl.int32,
        m: tl.int32,
        p: tl.int32,
        BLOCK_SIZE: tl.constexpr,
):
    """ Triton kernel for matrix multiplication and row dot product """
    pid = tl.program_id(axis=0)
    block_start_i = pid * BLOCK_SIZE
    offsets_i = block_start_i + tl.arange(0, BLOCK_SIZE)
    mask_i = offsets_i < n
    
    # Load data from A and B
    A = tl.load(A_ptr + offsets_i[:, None] * m)
    B = tl.load(B_ptr + offsets_i[None, :] * p)
    
    # Perform matrix multiplication
    C_block = tl.zeros((BLOCK_SIZE, p), dtype=tl.float32)
    for k in range(m):
        A_k = A[:, k]
        B_k = B[k]
        C_block += alpha * A_k[:, None] * B_k[None, :]
    
    # Add scaled version of C
    C_block += beta * tl.load(C_ptr + offsets_i[:, None] * p)
    
    # Store back to C
    tl.store(C_ptr + offsets_i[:, None] * p, C_block, mask=mask_i)
    
    # Compute dot product of the first two rows
    if pid == 0:
        row_0 = C_block[0]
        row_1 = C_block[1]
        dot_product = tl.dot(row_0, row_1)
        tl.store(C_ptr + n * p, dot_product)

def matrix_multiply_and_row_dot(
        A: torch.Tensor,
        B: torch.Tensor,
        alpha: float,
        beta: float,
        C: torch.Tensor
) -> torch.Tensor:
    """
    Wrapper function for matrix multiplication and row dot product
    :param A (torch.Tensor): First input matrix of shape `(n, m)`
    :param B (torch.Tensor): Second input matrix of shape `(m, p)`
    :param alpha (float): Scalar multiplier for the matrix-matrix product
    :param beta (float): Scalar multiplier for the input matrix `C`
    :param C (torch.Tensor): Output matrix of shape `(n, p)` where the results are added
    :return (torch.Tensor): Dot product of the first two rows of the updated matrix `C`
    """
    assert C.shape == (A.shape[0], B.shape[1]), "Shape mismatch"
    # Make sure C is contiguous
    if not C.is_contiguous():
        C = C.contiguous()
    
    # Call triton kernel
    grid = lambda meta: (triton.cdiv(A.shape[0], meta['BLOCK_SIZE']),)
    _matrix_multiply_and_row_dot_kernel[grid](
        A.contiguous().data_ptr(),
        B.contiguous().data_ptr(),
        C.contiguous().data_ptr(),
        alpha,
        beta,
        A.shape[0],
        A.shape[1],
        B.shape[1],
        BLOCK_SIZE=1024
    )
    
    # Return the dot product from C
    return C[-1]

# Example usage:
A = torch.randn(5, 3)
B = torch.randn(3, 4)
alpha = 2.0
beta = 1.0
C = torch.randn(5, 4)
result = matrix_multiply_and_row_dot(A, B, alpha, beta, C)
print("Dot Product:", result.item())
