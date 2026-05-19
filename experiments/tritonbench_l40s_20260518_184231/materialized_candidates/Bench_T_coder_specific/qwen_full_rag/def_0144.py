import torch
import triton
import triton.language as tl

@triton.jit
def _matrix_multiply_and_row_dot_forward(
        A_ptr,
        B_ptr,
        C_ptr,
        result_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        P: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_P: tl.constexpr,
):
    """ Triton kernel for matrix multiply and row dot forward """
    program_idx = tl.program_id(axis=0)
    # Compute the block row and column each program will go through
    block_row = program_idx // (N // BLOCK_SIZE_N)
    block_col = program_idx % (N // BLOCK_SIZE_N)
    # Compute the start of the block
    block_a_start = block_row * BLOCK_SIZE_M * N
    block_b_start = block_col * BLOCK_SIZE_N * P
    # Do the computation
    a_block_ptr = A_ptr + block_a_start + tl.arange(0, BLOCK_SIZE_M)[:, None] * N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    b_block_ptr = B_ptr + block_b_start + tl.arange(0, BLOCK_SIZE_N)[:, None] * P + tl.arange(0, BLOCK_SIZE_P)[None, :]
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_P), dtype=tl.float32)
    for m in range(0, M, BLOCK_SIZE_M):
        for p in range(0, P, BLOCK_SIZE_P):
            a = tl.load(a_block_ptr, mask=(m + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) & (tl.arange(0, BLOCK_SIZE_N)[None, :] < N))
            b = tl.load(b_block_ptr, mask=(tl.arange(0, BLOCK_SIZE_N)[:, None] < N) & (p + tl.arange(0, BLOCK_SIZE_P)[None, :] < P))
            acc += tl.dot(a, b)
            a_block_ptr += BLOCK_SIZE_M * N
            b_block_ptr += BLOCK_SIZE_N * P
    c = acc.to(C_ptr.dtype.element_ty)
    # Write back the result
    c_block_ptr = C_ptr + block_a_start + tl.arange(0, BLOCK_SIZE_M)[:, None] * P + tl.arange(0, BLOCK_SIZE_P)[None, :]
    tl.store(c_block_ptr, c, mask=(m + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) & (p + tl.arange(0, BLOCK_SIZE_P)[None, :] < P))

def _wrapper_matrix_multiply_and_row_dot_forward(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        beta: float
) -> torch.Tensor:
    """
    Wrapper function for matrix multiply and row dot forward triton kernel
    :param A (torch.Tensor): First input matrix of shape `(M, N)`
    :param B (torch.Tensor): Second input matrix of shape `(N, P)`
    :param C (torch.Tensor): Output matrix of shape `(M, P)`
    :param beta (float): Scalar multiplier for the input matrix `C`
    :return (torch.Tensor): Result of the dot product of the first two rows of C
    """
    # Constants
    M, N = A.shape
    N, P = B.shape
    # Make sure the matrices are contiguous
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    if not C.is_contiguous():
        C = C.contiguous()
    # Prepare output tensor
    result = torch.empty((1,), dtype=A.dtype, device=A.device)
    # Make C = beta * C
    C.mul_(beta)
    # Call the triton kernel
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']) * triton.cdiv(M, META['BLOCK_SIZE_M']),)
    _matrix_multiply_and_row_dot_forward[grid](
        A,
        B,
        C,
        result,
        M,
        N,
        P
    )
    # Return result
    return result
