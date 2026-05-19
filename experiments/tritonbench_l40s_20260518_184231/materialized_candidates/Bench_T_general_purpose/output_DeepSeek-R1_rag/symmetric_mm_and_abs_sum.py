import torch
import triton
import triton.language as tl

@triton.jit
def _symmetric_mm_abs_sum_kernel(
    A_ptr, C_ptr, alpha, beta, sum_ptr,
    n, m,
    stride_A_row, stride_A_col,
    stride_C_row, stride_C_col,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pairs = n * n
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_pairs

    for idx in tl.static_range(0, BLOCK_SIZE):
        if mask[idx]:
            linear_idx = block_start + idx
            i = linear_idx // n
            j = linear_idx % n

            dot = 0.0
            for k_offset in range(0, m, BLOCK_SIZE_K):
                k = k_offset + tl.arange(0, BLOCK_SIZE_K)
                k_mask = k < m

                a_i = tl.load(A_ptr + i * stride_A_row + k * stride_A_col, mask=k_mask, other=0.0)
                a_j = tl.load(A_ptr + j * stride_A_row + k * stride_A_col, mask=k_mask, other=0.0)

                dot += tl.sum(a_i * a_j)

            c_ij = tl.load(C_ptr + i * stride_C_row + j * stride_C_col)
            updated = alpha * dot + beta * c_ij
            abs_updated = tl.abs(updated)
            tl.atomic_add(sum_ptr, abs_updated)

def symmetric_mm_and_abs_sum(A: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    n, m = A.shape
    assert C.shape == (n, n), f"C must be of shape ({n}, {n}), but got {C.shape}"

    if not A.is_contiguous():
        A = A.contiguous()
    if not C.is_contiguous():
        C = C.contiguous()

    sum = torch.zeros(1, dtype=A.dtype, device=A.device)

    BLOCK_SIZE = 1024  # Number of threads per block
    BLOCK_SIZE_K = 64  # Elements of K processed per iteration

    grid = lambda meta: (triton.cdiv(n * n, meta['BLOCK_SIZE']),)

    _symmetric_mm_abs_sum_kernel[grid](
        A, C, alpha, beta, sum,
        n, m,
        A.stride(0), A.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return sum
