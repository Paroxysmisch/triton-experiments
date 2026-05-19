import torch
import triton
import triton.language as tl

@triton.jit
def _tril_mm_and_scale_kernel(
        A_ptr, B_ptr, C_ptr,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        TILE_K: tl.constexpr,
):
    # Map the program id to the row of A it should compute.
    pid = tl.program_id(axis=0)
    # This line does three things:
    # 1. Computes the starting position of our block in the overall result matrix C.
    # 2. Subsequently, each of the three loops will iterate over M, N, and K blocks respectively.
    # 3. Finally, we load A and B pointers to the beginning of their respective blocks.
    r = pid // N
    c = pid % N
    # We only want to handle valid blocks for A and B
    mask_a = r * N + tl.arange(0, TILE_K) < M * K
    mask_b = c * N + tl.arange(0, TILE_K) < K * N

    off_k = tl.arange(0, TILE_K)
    off_rm = r * N + off_k
    off_cn = c * N + off_k

    A_ptrs = A_ptr + off_rm[:, None] + off_k[None, :]
    B_ptrs = B_ptr + off_k[:, None] + off_cn[None, :]

    acc = tl.zeros((TILE_K, TILE_K), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, TILE_K)):
        a = tl.load(A_ptrs, mask=mask_a, other=0.)
        b = tl.load(B_ptrs, mask=mask_b, other=0.)

        acc += tl.dot(a, b)
        A_ptrs += TILE_K
        B_ptrs += TILE_K

    acc = tl.sum(acc, axis=0)[None, :]

    C = acc.to(C_ptr.dtype.element_ty)

    mask_c = r * N + tl.arange(0, TILE_K) < M * N
    C_ptrs = C_ptr + off_rm[:, None] + off_cn[None, :]
    tl.store(C_ptrs, C, mask=mask_c)

def tril_mm_and_scale_triton(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """
    Wrapper function for TRITON lower triangular matrix multiplication and scaling
    :param A (torch.Tensor): A 2D tensor of shape (n, n)
    :param B (torch.Tensor): A 2D tensor of shape (n, p)
    :param alpha (float): Scaling factor for the intermediate product
    :param beta (float): Scaling factor for the final output
    :return (torch.Tensor): Output tensor after scaling and multiplication
    """

    assert A.shape[0] == A.shape[1], "Matrix A must be square"
    assert A.shape[1] == B.shape[0], f"Incompatible dimensions, {A.shape} - {B.shape}"

    M, K = A.shape
    _, N = B.shape

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    def grid(META):
        return (triton.cdiv(M * N, META["TILE_K"] ** 2), )

    tril_A = torch.tril(A)
    _tril_mm_and_scale_kernel[grid](
        tril_A, B, C,
        M, N, K,
        TILE_K=4,
    )
    C *= beta
    return C
