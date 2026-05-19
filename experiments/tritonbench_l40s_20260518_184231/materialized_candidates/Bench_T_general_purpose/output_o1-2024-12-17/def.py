import triton
import triton.language as tl

@triton.jit
def _lu_decomposition_kernel(A_ptr, L_ptr, U_ptr, P_ptr, stride_a_row, stride_a_col, stride_l_row, stride_l_col, stride_u_row, stride_u_col, stride_p, n, pivot, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * BLOCK
    offsets = tl.arange(0, BLOCK)
    mask = offsets + row_start < n
    for k in range(n):
        if pivot != 0:
            max_idx = k
            max_val = tl.abs(tl.load(A_ptr + (row_start + k) * stride_a_row + k * stride_a_col, mask=mask, other=0))
            for r in range(k+1, n):
                val = tl.abs(tl.load(A_ptr + (row_start + r) * stride_a_row + k * stride_a_col, mask=mask, other=0))
                cond = val > max_val
                max_val = tl.where(cond, val, max_val)
                max_idx = tl.where(cond, r, max_idx)
            if max_idx != k:
                for c in range(n):
                    tmp1 = tl.load(A_ptr + (row_start + k) * stride_a_row + c * stride_a_col, mask=mask, other=0)
                    tmp2 = tl.load(A_ptr + (row_start + max_idx) * stride_a_row + c * stride_a_col, mask=mask, other=0)
                    tl.store(A_ptr + (row_start + k) * stride_a_row + c * stride_a_col, tmp2, mask=mask)
                    tl.store(A_ptr + (row_start + max_idx) * stride_a_row + c * stride_a_col, tmp1, mask=mask)
                tmpi = tl.load(P_ptr + row_start + k, mask=mask, other=0)
                tmpj = tl.load(P_ptr + row_start + max_idx, mask=mask, other=0)
                tl.store(P_ptr + row_start + k, tmpj, mask=mask)
                tl.store(P_ptr + row_start + max_idx, tmpi, mask=mask)
        pivot_val = tl.load(A_ptr + (row_start + k) * stride_a_row + k * stride_a_col, mask=mask, other=0)
        for r in range(k+1, n):
            val = tl.load(A_ptr + (row_start + r) * stride_a_row + k * stride_a_col, mask=mask, other=0)
            fac = val / pivot_val
            tl.store(A_ptr + (row_start + r) * stride_a_row + k * stride_a_col, fac, mask=mask)
            for c in range(k+1, n):
                ac = tl.load(A_ptr + (row_start + r) * stride_a_row + c * stride_a_col, mask=mask, other=0)
                pivac = tl.load(A_ptr + (row_start + k) * stride_a_row + c * stride_a_col, mask=mask, other=0)
                tl.store(A_ptr + (row_start + r) * stride_a_row + c * stride_a_col, ac - fac * pivac, mask=mask)

    for i in range(n):
        for j in range(n):
            val = tl.load(A_ptr + (row_start + i) * stride_a_row + j * stride_a_col, mask=mask, other=0)
            if j < i:
                tl.store(L_ptr + (row_start + i) * stride_l_row + j * stride_l_col, val, mask=mask)
                tl.store(U_ptr + (row_start + i) * stride_u_row + j * stride_u_col, 0.0, mask=mask)
            else:
                tl.store(U_ptr + (row_start + i) * stride_u_row + j * stride_u_col, val, mask=mask)
                if j == i:
                    tl.store(L_ptr + (row_start + i) * stride_l_row + j * stride_l_col, 1.0, mask=mask)
                else:
                    tl.store(L_ptr + (row_start + i) * stride_l_row + j * stride_l_col, 0.0, mask=mask)

@triton.jit
def _forward_substitution_kernel(L_ptr, B_ptr, P_ptr, stride_l_row, stride_l_col, stride_b_row, stride_b_col, stride_p, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * BLOCK
    offsets = tl.arange(0, BLOCK)
    mask = offsets + row_start < n
    for i in range(n):
        pi = tl.load(P_ptr + row_start + i, mask=mask, other=0)
        tmp = tl.load(B_ptr + (row_start + i) * stride_b_row, mask=mask, other=0)
        pval = tl.load(B_ptr + (row_start + pi) * stride_b_row, mask=mask, other=tmp)
        tl.store(B_ptr + (row_start + i) * stride_b_row, pval, mask=mask)
        lii = tl.load(L_ptr + (row_start + i) * stride_l_row + i * stride_l_col, mask=mask, other=1.0)
        for j in range(i):
            lij = tl.load(L_ptr + (row_start + i) * stride_l_row + j * stride_l_col, mask=mask, other=0)
            bj = tl.load(B_ptr + (row_start + j) * stride_b_row, mask=mask, other=0)
            tmp -= lij * bj
        tmp = tmp / lii
        tl.store(B_ptr + (row_start + i) * stride_b_row, tmp, mask=mask)

@triton.jit
def _backward_substitution_kernel(U_ptr, B_ptr, stride_u_row, stride_u_col, stride_b_row, stride_b_col, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * BLOCK
    offsets = tl.arange(0, BLOCK)
    mask = offsets + row_start < n
    for i in range(n-1, -1, -1):
        tmp = tl.load(B_ptr + (row_start + i) * stride_b_row, mask=mask, other=0)
        uii = tl.load(U_ptr + (row_start + i) * stride_u_row + i * stride_u_col, mask=mask, other=1.0)
        for j in range(i+1, n):
            uij = tl.load(U_ptr + (row_start + i) * stride_u_row + j * stride_u_col, mask=mask, other=0)
            bj = tl.load(B_ptr + (row_start + j) * stride_b_row, mask=mask, other=0)
            tmp -= uij * bj
        tmp = tmp / uii
        tl.store(B_ptr + (row_start + i) * stride_b_row, tmp, mask=mask)

def solve_multiple_lu(A, Bs, *, pivot=True, out=None) -> "Tensor":
    import torch
    if out is None:
        out = torch.empty_like(Bs)
    orig_shape = A.shape
    batch_dims = orig_shape[:-2]
    n = A.shape[-1]
    k = Bs.shape[-1]
    A_reshaped = A.reshape(-1, n, n)
    Bs_reshaped = Bs.reshape(-1, n, k)
    num_batches = A_reshaped.shape[0]
    L = torch.empty_like(A_reshaped)
    U = torch.empty_like(A_reshaped)
    P = torch.arange(n, device=A.device).unsqueeze(0).expand(num_batches, -1).clone()
    BLOCK = 1
    grids = lambda meta: (num_batches,)
    stride_a_row = A_reshaped.stride(1)
    stride_a_col = A_reshaped.stride(2)
    stride_l_row = L.stride(1)
    stride_l_col = L.stride(2)
    stride_u_row = U.stride(1)
    stride_u_col = U.stride(2)
    stride_p = P.stride(1)
    for i in range(k):
        b_slice = Bs_reshaped[..., i].contiguous()
        _A = A_reshaped
        _B = b_slice
        l_stride_row = L.stride(1)
        l_stride_col = L.stride(2)
        u_stride_row = U.stride(1)
        u_stride_col = U.stride(2)
        if i == 0:
            _lu_decomposition_kernel[grids](
                _A, L, U, P,
                stride_a_row, stride_a_col,
                stride_l_row, stride_l_col,
                stride_u_row, stride_u_col,
                stride_p,
                n, int(pivot), BLOCK=BLOCK
            )
        _forward_substitution_kernel[grids](
            L, _B, P,
            l_stride_row, l_stride_col,
            _B.stride(0), 1,
            stride_p, n, BLOCK=BLOCK
        )
        _backward_substitution_kernel[grids](
            U, _B,
            u_stride_row, u_stride_col,
            _B.stride(0), 1,
            n, BLOCK=BLOCK
        )
        out[..., i].copy_(_B.view(*batch_dims, n))
    return out
