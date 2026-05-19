import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64}),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128}),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 256}),
    ],
    key=['N', 'M']
)
@triton.jit
def mv_kernel(
    A_ptr, B_ptr, C_ptr,
    N, M,
    stride_an, stride_am,
    stride_b, stride_c,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr
):
    pid = tl.program_id(0)
    row_off = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    accum = tl.zeros([BLOCK_N], dtype=tl.float32)
    cols = tl.arange(0, BLOCK_M)

    for i in range(0, M, BLOCK_M):
        a_ptrs = A_ptr + row_off[:, None] * stride_an + (i + cols)[None, :] * stride_am
        b_ptrs = B_ptr + (i + cols) * stride_b
        mask_a = (row_off[:, None] < N) & (i + cols[None, :] < M)
        mask_b = (i + cols < M)
        a = tl.load(a_ptrs, mask=mask_a, other=0)
        b = tl.load(b_ptrs, mask=mask_b, other=0)
        accum += tl.sum(a * b, axis=1)

    c_ptrs = C_ptr + row_off * stride_c
    mask_c = row_off < N
    tl.store(c_ptrs, accum, mask=mask_c)

def mv(A, B, C):
    import triton
    assert A.is_contiguous() and B.is_contiguous() and C.is_contiguous()
    N, M = A.shape
    assert B.shape[0] == M and C.shape[0] == N
    grid = ((N + 128 - 1) // 128,)
    mv_kernel[grid](
        A, B, C,
        N, M,
        A.stride(0), A.stride(1),
        B.stride(0), C.stride(0)
    )
