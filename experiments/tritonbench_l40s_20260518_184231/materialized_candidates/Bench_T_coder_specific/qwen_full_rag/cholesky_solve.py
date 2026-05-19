import torch
import triton
import triton.language as tl

real_dtype_to_complex_dtype = {
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}

@triton.jit
def _cholesky_fwd_triton(
    A,  # Pointer to the input matrix
    L,  # Pointer to the output matrix
    stride_za,  # Stride for accessing elements in A
    stride_zm,  # Stride for accessing elements in A along the m dimension
    stride_zn,  # Stride for accessing elements in A along the n dimension
    stride_zl,  # Stride for accessing elements in L
    stride_zlm,  # Stride for accessing elements in L along the m dimension
    stride_zln,  # Stride for accessing elements in L along the n dimension
    batch,  # Batch size
    n,  # Number of columns in the input matrix
    min_n,  # Minimum value of n
    one_by_sqrt_two: tl.constexpr,  # Constant expression for 1/sqrt(2)
    BLOCK_SIZE_M: tl.constexpr,  # Block size for the m dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for the n dimension
):
    program_idx = tl.program_id(axis=0)
    program_in_group_count = tl.num_programs(0)
    for z in range(batch):
        if program_idx < program_in_group_count:
            rm = program_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
            rn = tl.arange(0, BLOCK_SIZE_N)[None, :]
            A += z * stride_za
            L += z * stride_zl
            a = tl.load(A + rm * stride_zm + rn * stride_zn, mask=(rm < n) & (rn < n), other=0.0)
            a = a + a.T
            a = tl.where(rm >= rn, a, 0.0)
            m = tl.min(rm, rn)
            d = tl.sum(a / (one_by_sqrt_two * 2.0) ** m, axis=0)
            l = tl.sqrt(d)
            tl.store(L + rm * stride_zlm + rn * stride_zln, l, mask=(rm < n) & (rn < n))
            program_idx += program_in_group_count

@triton.jit
def _cholesky_fwd_triton_special(
    A,  # Pointer to the input matrix
    L,  # Pointer to the output matrix
    stride_za,  # Stride for accessing elements in A
    stride_zm,  # Stride for accessing elements in A along the m dimension
    stride_zn,  # Stride for accessing elements in A along the n dimension
    stride_zl,  # Stride for accessing elements in L
    stride_zlm,  # Stride for accessing elements in L along the m dimension
    stride_zln,  # Stride for accessing elements in L along the n dimension
    batch,  # Batch size
    n,  # Number of columns in the input matrix
    min_n,  # Minimum value of n
    one_by_sqrt_two: tl.constexpr,  # Constant expression for 1/sqrt(2)
    BLOCK_SIZE_M: tl.constexpr,  # Block size for the m dimension
    BLOCK_SIZE_N: tl.constexpr,  # Block size for the n dimension
):
    for z in range(batch):
        A += z * stride_za
        L += z * stride_zl
        a = tl.load(A + 0 * stride_zm + 0 * stride_zn, mask=True, other=0.0)
        tl.store(L + 0 * stride_zlm + 0 * stride_zln, a, mask=True)

def _cholesky_fwd_special(A, L):
    batch, n, _ = A.shape
    min_n = min(math.ceil(n ** 0.5), 8)
    BLOCK_SIZE_M = triton.next_power_of_2(min_n)
    BLOCK_SIZE_N = 8
    num_warps = 1
    num_stages = 4
    _cholesky_fwd_triton_special[(batch,)](
        A,
        L,
        stride_za=A.stride(0),
        stride_zm=A.stride(1),
        stride_zn=A.stride(2),
        stride_zl=L.stride(0),
        stride_zlm=L.stride(1),
        stride_zln=L.stride(2),
        batch=batch,
        n=n,
        min_n=min_n,
        one_by_sqrt_two=2.0**-0.5,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

def _cholesky_fwd(A, L):
    batch, n, _ = A.shape
    min_n = min(math.ceil(n ** 0.5), 8)
    BLOCK_SIZE_M = triton.next_power_of_2(min_n)
    BLOCK_SIZE_N = 8
    num_warps = 1
    num_stages = 4
    if min_n > 8:
        _cholesky_fwd_special[(batch,)](
            A,
            L,
            stride_za=A.stride(0),
            stride_zm=A.stride(1),
            stride_zn=A.stride(2),
            stride_zl=L.stride(0),
            stride_zlm=L.stride(1),
            stride_zln=L.stride(2),
            batch=batch,
            n=n,
            min_n=min_n,
            one_by_sqrt_two=2.0**-0.5,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return
    _cholesky_fwd_triton[(batch, triton.cdiv(n, BLOCK_SIZE_M),)](
        A,
        L,
        stride_za=A.stride(0),
        stride_zm=A.stride(1),
        stride_zn=A.stride(2),
        stride_zl=L.stride(0),
        stride_zlm=L.stride(1),
        stride_zln=L.stride(2),
        batch=batch,
        n=n,
        min_n=min_n,
        one_by_sqrt_two=2.0**-0.5,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
