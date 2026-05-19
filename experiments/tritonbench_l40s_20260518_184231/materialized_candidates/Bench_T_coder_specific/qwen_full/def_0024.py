import torch
import triton
import triton.language as tl

@triton.jit
def _lu_solve_kernel(
    A_ptr, b_ptr, x_ptr, n_ptr, batch, m, n, nrhs, lda, ldb, ldc, n_block, **meta
):
    pid = tl.program_id(0)
    if pid < batch:
        n_cur = tl.load(n_ptr + pid)
        if n_cur <= 0:
            return
        m_id = tl.arange(0, n_block)
        b_id = tl.arange(0, n_block)
        A_offs = pid * lda + m_id[:, None] * lda + b_id[None, :] * 1
        b_offs = pid * ldb + m_id[:, None] * ldb + b_id[None, :] * nrhs
        x_offs = pid * ldc + m_id[:, None] * ldc + b_id[None, :] * nrhs
        A_ptrs = A_ptr + A_offs
        b_ptrs = b_ptr + b_offs
        x_ptrs = x_ptr + x_offs
        r = tl.zeros([n_block, n_block], dtype=tl.float32)
        for k in range(n_cur):
            r += tl.dot(tl.load(A_ptrs), tl.load(b_ptrs), allow_tf32=False)
            A_ptrs += 1
            b_ptrs += 1
        tl.store(x_ptrs, r.to(x_ptr.dtype.element_ty), mask=b_id[None, :] < nrhs)
