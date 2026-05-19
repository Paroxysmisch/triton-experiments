import triton
import triton.language as tl

@triton.jit
def qr_decomposition_kernel(
    A_ptr, Q_ptr, R_ptr, M, N, stride_am, stride_an, stride_qm, stride_qn, stride_rm, stride_rn, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets_m = block_start + tl.arange(0, BLOCK_SIZE)
    offsets_n = tl.arange(0, BLOCK_SIZE)

    mask = (offsets_m < M) & (offsets_n < N)
    A = tl.load(A_ptr + offsets_m[:, None] * stride_am + offsets_n[None, :] * stride_an, mask=mask, other=0.0)

    Q = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    R = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for i in range(BLOCK_SIZE):
        for j in range(i, BLOCK_SIZE):
            R[i, j] = tl.sum(Q[:, i] * A[:, j])
        for j in range(i, BLOCK_SIZE):
            A[:, j] -= Q[:, i] * R[i, j]
        norm = tl.sqrt(tl.sum(A[:, i] * A[:, i]))
        Q[:, i] = A[:, i] / norm
        R[i, i] = norm

    tl.store(Q_ptr + offsets_m[:, None] * stride_qm + offsets_n[None, :] * stride_qn, Q, mask=mask)
    tl.store(R_ptr + offsets_m[:, None] * stride_rm + offsets_n[None, :] * stride_rn, R, mask=mask)

@triton.jit
def determinant_kernel(
    R_ptr, det_ptr, M, N, stride_rm, stride_rn, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets_m = block_start + tl.arange(0, BLOCK_SIZE)
    offsets_n = tl.arange(0, BLOCK_SIZE)

    mask = (offsets_m < M) & (offsets_n < N)
    R = tl.load(R_ptr + offsets_m[:, None] * stride_rm + offsets_n[None, :] * stride_rn, mask=mask, other=0.0)

    det = 1.0
    for i in range(BLOCK_SIZE):
        det *= R[i, i]

    tl.atomic_add(det_ptr, det)

### Wrapper Function
