import torch
import triton
import triton.language as tl

@triton.jit
def lu_no_pivot_kernel(
    a_ptr, l_ptr, u_ptr,
    m, n,
    stride_a_batch, stride_a_row, stride_a_col,
    stride_l_batch, stride_l_row, stride_l_col,
    stride_u_batch, stride_u_row, stride_u_col,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    
    a_offset = pid_batch * stride_a_batch
    l_offset = pid_batch * stride_l_batch
    u_offset = pid_batch * stride_u_batch

    # Copy A to U and initialize L as identity
    for i in range(m):
        for j in range(n):
            u_p = u_offset + i * stride_u_row + j * stride_u_col
            a_p = a_offset + i * stride_a_row + j * stride_a_col
            val = tl.load(a_ptr + a_p)
            tl.store(u_ptr + u_p, val)
    for i in range(m):
        for j in range(m):
            l_p = l_offset + i * stride_l_row + j * stride_l_col
            val = 1.0 if i == j else 0.0
            tl.store(l_ptr + l_p, val)

    for k in range(min(m, n)):
        u_kk_p = u_offset + k * stride_u_row + k * stride_u_col
        u_kk = tl.load(u_ptr + u_kk_p)
        for i in range(k + 1, m):
            l_ik_p = l_offset + i * stride_l_row + k * stride_l_col
            u_ik_p = u_offset + i * stride_u_row + k * stride_u_col
            u_ik = tl.load(u_ptr + u_ik_p)
            l_ik = u_ik / u_kk
            tl.store(l_ptr + l_ik_p, l_ik)
            tl.store(u_ptr + u_ik_p, 0.0)
            for j in range(k + 1, n):
                u_ij_p = u_offset + i * stride_u_row + j * stride_u_col
                u_kj_p = u_offset + k * stride_u_row + j * stride_u_col
                u_ij = tl.load(u_ptr + u_ij_p)
                u_kj = tl.load(u_ptr + u_kj_p)
                u_ij_updated = u_ij - l_ik * u_kj
                tl.store(u_ptr + u_ij_p, u_ij_updated)

def lu(A, *, pivot=True, out=None):
    assert A.dim() >= 2, "A must have at least 2 dimensions"
    m, n = A.shape[-2], A.shape[-1]
    batch_dims = A.shape[:-2]
    device = A.device

    if pivot:
        lu_tensor, pivots = torch.lu(A, get_infos=False)
        P, L, U = torch.lu_unpack(lu_tensor, pivots)
        return (P, L, U)
    else:
        if not A.is_cuda:
            raise NotImplementedError("LU without pivoting is only supported on GPU")
        
        L = torch.empty(*batch_dims, m, m, dtype=A.dtype, device=device)
        U = torch.empty_like(A)
        num_batches = 1
        for dim in batch_dims:
            num_batches *= dim

        BLOCK_SIZE = 16  # Tune this based on hardware

        lu_no_pivot_kernel[(num_batches,)](
            A, L, U,
            m, n,
            A.stride(-3) if A.dim() >=3 else 0,
            A.stride(-2), A.stride(-1),
            L.stride(-3) if L.dim() >=3 else 0,
            L.stride(-2), L.stride(-1),
            U.stride(-3) if U.dim() >=3 else 0,
            U.stride(-2), U.stride(-1),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        P = torch.tensor([], dtype=A.dtype, device=device)
        return (P, L, U)
