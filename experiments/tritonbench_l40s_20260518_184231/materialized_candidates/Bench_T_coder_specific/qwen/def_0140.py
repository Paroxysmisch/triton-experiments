import torch

def tril_mm_and_scale(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    assert A.dim() == 2, "Matrix A must be 2D"
    assert B.dim() == 2, "Matrix B must be 2D"
    assert A.shape[0] == A.shape[1], "Matrix A must be square"
    assert A.shape[0] == B.shape[0], "Number of rows in A must match number of rows in B"

    N, P = A.shape[0], B.shape[1]
    C = torch.empty((N, P), device=A.device, dtype=A.dtype)

    block_size = 32
    grid = ((N + block_size - 1) // block_size, (P + block_size - 1) // block_size)

    tril_mm_kernel[grid](A.data_ptr(), B.data_ptr(), C.data_ptr(),
                         A.stride(0), A.stride(1),
                         B.stride(0), B.stride(1),
                         C.stride(0), C.stride(1),
                         N, P, alpha, beta,
                         BLOCK_SIZE=block_size)

    return C
