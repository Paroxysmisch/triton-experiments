import torch
import triton
import triton.language as tl

@triton.jit
def _tril_mm_and_scale_kernel(
    A_ptr,  # pointer to A
    B_ptr,  # pointer to B
    C_ptr,  # pointer to output (C)
    n,      # dimension n
    p,      # dimension p
    alpha,  # scaling factor alpha
    beta,   # scaling factor beta
    BLOCK_SIZE: tl.constexpr
):
    """
    Naive Triton kernel for:
      C = beta * [ alpha * tril(A) @ B ]
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Avoid out-of-bounds
    mask = offsets < (n * p)
    offsets = tl.where(mask, offsets, 0)
    # Compute row (i), col (j) from flattened index
    i = offsets // p
    j = offsets % p

    acc = 0.0
    # Dot product: sum_{k=0..n-1}[A(i, k)*B(k, j)] subject to i >= k
    # row-major indexing: A[i, k] => A_ptr + i*n + k; B[k, j] => B_ptr + k*p + j
    for k in range(1024):  # limit for safety; adjustable if n <= 1024
        # Only accumulate if k < n; also check triangular condition i >= k
        valid = (k < n) & (i >= k) & mask
        a_val = tl.where(valid, tl.load(A_ptr + i * n + k), 0.0)
        b_val = tl.where(valid, tl.load(B_ptr + k * p + j), 0.0)
        acc += a_val * b_val

    # Scale by alpha, then scale by beta => alpha * beta
    acc = acc * alpha * beta
    # Store result
    c_offset = i * p + j
    tl.store(C_ptr + c_offset, acc, mask=mask)


def tril_mm_and_scale(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """
    Performs: 
      1) B = alpha * (tril(A) @ B)
      2) C = beta * B

    Parameters:
      A (Tensor): shape (n, n)
      B (Tensor): shape (n, p)
      alpha (float)
      beta (float)
    Returns:
      C (Tensor): shape (n, p)
    """
    # Ensure contiguous
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()

    n, nA = A.shape
    nB, p = B.shape
    assert n == nA, "A must be n x n"
    assert nB == n, "B must be n x p"
    # Prepare output
    C = torch.empty((n, p), dtype=A.dtype, device=A.device)
    # Grid/block launch
    numel = n * p
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    # Launch kernel
    _tril_mm_and_scale_kernel[grid](
        A, B, C,
        n, p,
        alpha, beta,
        BLOCK_SIZE=1024
    )
    return C
