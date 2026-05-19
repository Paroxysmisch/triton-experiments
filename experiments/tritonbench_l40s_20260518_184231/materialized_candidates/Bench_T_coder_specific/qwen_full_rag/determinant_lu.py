import torch
import triton
import triton.language as tl
from flag_gems.utils.shape_utils import volume

# Triton kernel for computing the determinant of a matrix using LU decomposition
@triton.autotune(
    configs=[
        triton.Config({"N_BLOCK": 256, "NK_BLOCK": 64}, num_warps=8),
        triton.Config({"N_BLOCK": 256, "NK_BLOCK": 128}, num_warps=8),
        triton.Config({"N_BLOCK": 128, "NK_BLOCK": 32}, num_warps=8),
        triton.Config({"N_BLOCK": 128, "NK_BLOCK": 64}, num_warps=8),
        triton.Config({"N_BLOCK": 128, "NK_BLOCK": 128}, num_warps=8),
        triton.Config({"N_BLOCK": 256, "NK_BLOCK": 32}, num_warps=8),
        triton.Config({"N_BLOCK": 512, "NK_BLOCK": 32}, num_warps=8),
        triton.Config({"N_BLOCK": 256, "NK_BLOCK": 64}, num_warps=4),
        triton.Config({"N_BLOCK": 256, "NK_BLOCK": 128}, num_warps=4),
        triton.Config({"N_BLOCK": 128, "NK_BLOCK": 32}, num_warps=4),
        triton.Config({"N_BLOCK": 128, "NK_BLOCK": 64}, num_warps=4),
        triton.Config({"N_BLOCK": 128, "NK_BLOCK": 128}, num_warps=4),
        triton.Config({"N_BLOCK": 256, "NK_BLOCK": 32}, num_warps=4),
        triton.Config({"N_BLOCK": 512, "NK_BLOCK": 32}, num_warps=4),
    ],
    key=["n"],
)
@triton.jit
def _determinant_lu_kernel(
    N,
    A_ptr,
    stride_ap,
    P_ptr,
    det_ptr,
    TMP_ptr,
    n_batches,
    n,
    CACHE_KEY_BATCH,
    CACHE_KEY_N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    N_BLOCK: tl.constexpr,
    NK_BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    # compute offset
    offset_pidm = pid_m * BLOCK_SIZE_M
    m_offsets = offset_pidm + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = tl.arange(0, N_BLOCK)
    uk_offsets = tl.arange(0, NK_BLOCK)
    # initialize
    mask_m = m_offsets < n
    mask_n = n_offsets < N
    # pointer check
    A_ptr += (
        pid_batch * stride_ap + (m_offsets[:, None] * N + n_offsets[None, :]) * 2
    )  # 2 is for cfloat
    tmp_ptr = TMP_ptr + (pid_batch * n + n_offsets)
    det = tl.full([BLOCK_SIZE_M], 1.0, dtype=tl.float32)
    # do block
    for k in range(0, tl.cdiv(N, N_BLOCK)):
        _A = tl.load(
            A_ptr, mask=(mask_m[:, None] & (n_offsets[None, :] < N - k)), eviction_policy="evict_last"
        ).to(tl.float32)
        Uk = tl.view(_A, (N_BLOCK, -1))
        if P_ptr is not None:
            P_ptr += pid_batch * n + n_offsets
            p = tl.load(P_ptr, mask=mask_n, eviction_policy="evict_first")
            p = p.to(tl.float32)[None, :]
            _sign = (p % 2 == 0).to(tl.int32)
            sign = 1.0 - 2.0 * _sign
            det *= sign
            # print(sign, "!!!!!!!!!")
        r = tl.nvml.trtri(Uk, offset=0, uplo="U", unit_diag=False)
        if k == 0:
            U_ptr = tmp_ptr
            tl.store(U_ptr, r, mask=mask_n)
        else:
            # this is stupid, but i can't get the address of a block inside a tensor..
            last_u_ptr = U_ptr + (N_BLOCK * 2)  # 2 is for cfloat
            tl.store(last_u_ptr, r, mask=mask_n)
            U_ptr += N_BLOCK * 2
        A_ptr += 2 * N * BLOCK_SIZE_M
    # compute
    ldt = tl.log(det)
    # reduce
    ldt_r = tl.broadcast_to(ldt, (n,))
    _sum = tl.sum(ldt_r, axis=0)
    sum_ = tl.exp(_sum)
    tl.store(det_ptr, sum_)


# Wrapper function for calling the Triton kernel
def determinant_lu(A, *, pivot=True, out=None):
    assert A.is_contiguous(), "Input must be contiguous"
    n_dims = A.dim()
    n = A.size(-1)
    assert A.shape[-2:] == (n, n), f"Last 2 dimensoin must be equal to {n}"

    n_batches = A.numel() // volume(A.shape[:-2])
    if n_batches == 0:
        n_batches = 1

    if out is None:
        out = torch.empty(
            (*A.shape[:-2], 1), dtype=A.dtype, device=A.device, requires_grad=False
        )
    else:
        assert list(out.shape) == list(A.shape)[:-2] + [1]
        assert out.is_contiguous()

    # allocates temporary buffer needed for the computation
    byte_size = volume(A.shape) * A.element_size()
    grid = lambda META: (triton.cdiv(n, META["BLOCK_SIZE_M"]), n_batches)
    TMP = torch.empty((n_batches, n * 2), dtype=A.dtype, device=A.device)  # 2 is for cfloat
    P = torch.arange(0, n, dtype=torch.int32, device=A.device)  # placeholder

    # enqueue kernel
    _determinant_lu_kernel[grid](
        n,
        A,
        A.stride(-2),
        P if pivot else None,
        out,
        TMP,
        n_batches,
        n,
        n,
        n,
        BLOCK_SIZE_M=n,
        NUM_SM=48,
    )

    return out
