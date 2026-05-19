import torch
import triton
import triton.language as tl
from flag_gems.utils.shape_utils import volume


# Kernel function: Calculates the rank of a matrix using QR decomposition.
@triton.jit
def _rank_qr_kernel(
    mat_ptr,
    tau_ptr,
    m,
    n,
    k,
    stride_mat_m,
    stride_mat_n,
    stride_tau_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    mat_tile_ptr = mat_ptr + pid_x * BLOCK_M * stride_mat_m + pid_y * BLOCK_N * stride_mat_n

    mat = tl.load(mat_tile_ptr, mask=(BLOCK_M, BLOCK_N), other=0.0)
    trans_mat = tl.trans(mat)

    q, r, tau = tl.qr(trans_mat, mode="reduced")

    tau_store_ptr = tau_ptr + pid_x * k * stride_tau_k + pid_y * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(tau_store_ptr, tau, mask=(k, BLOCK_N))


def gemm_based_rank(input_qr, eps=1e-8):
    _, M, N = input_qr.shape
    k = min(M, N)
    q, r = input_qr[:, :, :k], input_qr[:, :, k:]
    tau = torch.zeros((M, k), device=r.device, dtype=r.dtype)
    num_copies = volume(q) // (q.shape[0] * q.shape[1])

    if num_copies > 1:
        q_expanded = q.unsqueeze(-1).expand(-1, -1, num_copies)
        r_expanded = r.unsqueeze(1).expand(-1, num_copies, -1)
        r_copy_stride = r.stride(0), *r.stride()[1:]
        qr_expanded = torch.cat([q_expanded, r_expanded], dim=-1)
        gemm_ranks = gemm_based_rank(qr_expanded, eps=eps)
        gemm_ranks = torch.split(gemm_ranks, gemm_ranks.size(0) // num_copies, dim=0)
        rank = max(*gemm_ranks)
    else:
        q, r, _ = torch.linalg.gesvd(q, full_matrices=False, compute_uv=True, out=(q, r))
        diag_r, _ = torch.max(torch.abs(r), dim=1)
        rank = torch.sum(diag_r > eps).item()
    torch.diagonal(tau, 0, -1, -2)[:] = tau
    info = (rank, q, r, tau)
    return info


def fused_qr_solve_wrapper(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert (
        len(A.shape) == 2 and len(b.shape) <= 3
    ), "Input matrices must be 2D, but got {} and {}".format(A.shape, b.shape)
    assert (
        A.is_contiguous() and b.is_contiguous()
    ), "Both inputs must be contiguous, but got {} and {}".format(A.is_contiguous(), b.is_contiguous())
    assert (
        A.shape[-1] <= A.shape[-2]
    ), "Last dimension must be smaller than second last, but got {}".format(A.shape)
    if b.dim() == 2:
        batch_shape = b.shape[:-1]
        b = b.reshape(-1, b.shape[-1])
    else:
        batch_shape = b.shape
    A = A.expand(b.shape[-2], -1, -1)
    n, m, k = A.shape[-2], A.shape[-1], b.shape[-1]
    output = torch.empty([n, m, k], dtype=A.dtype, device=A.device)
    handle = get_tmgemm_handle(m=m, n=n, k=k, batch_shape=batch_shape)
    grid_fn = lambda meta: [n // meta["BLOCK_N"], m // meta["BLOCK_M"]]
    _fused_qr_solve_kernel[grid_fn](
        A,
        b,
        output,
        m,
        k,
        stride_am=n,
        stride_ak=k,
        stride_bk=k,
        stride_bn=n,
        stride_ok=n,
        stride_om=k,
        BLOCK_M=16,
        BLOCK_N=16,
        SPLIT_K=1,
    )
    output = output.squeeze(0)
    if batch_shape:
        output = output.reshape(batch_shape + (-1,))
    return output


def get_tmgemm_handle(m, n, k, batch_shape):
    key = (m, n, k, *batch_shape)
    if key in g_handles:
        return g_handles[key]
    else:
        handle = TorchTimedGEMMHandle(m, n, k, batch_shape)
        g_handles[key] = handle
        return handle


class TorchTimedGEMMHandle:
    def __init__(self, m, n, k, batch_shape):
        self.batch_dims = batch_shape
        self.m = m
        self.n = n
        self.k = k
        self.raw_handle = None

    def call(self, a, b, output=None):
        if output is None:
            output = torch.empty(
                (*self.batch_dims, self.m, self.n),
                device=a.device,
                dtype=torch.get_autocast_gpu_dtype(),
            )
        else:
            assert output.shape == (*self.batch_dims, self.m, self.n)
        if a.stride(-1) != 1 and a.stride(-2) != 1:
            a = a.contiguous()
        if b.stride(-1) != 1 and b.stride(-2) != 1:
            b = b.contiguous()
        assert a.stride(-1) == 1 or a.stride(-2) == 1
        assert b.stride(-1) == 1 or b.stride(-2) == 1
        if self.raw_handle is None:
            self.raw_handle = _tmgemm_handle_init(self.m, self.n, self.k, len(self.batch_dims))
        _tmgemm_run(
            self.raw_handle,
            a,
            b,
            output,
            list(self.batch_dims),
            torch.cuda.current_device(),
        )
        return output


g_handles = dict()


# Kernel function: Fused QR solve operation.
@triton.autotune(configs=_get_config(), key=["n", "k"])
@triton.jit
def _fused_qr_solve_kernel(
    mat_ptr,
    rhs_ptr,
    output_ptr,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ok,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_sk = tl.program_id(axis=1)
    nwarps = 4

    rm = pid % nwarps
    rn = pid // nwarps

    rk = pid_sk

    offs_m = rm * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = rn * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < n
    n_mask = offs_n < k

    a_ptrs = mat_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_ak)
    ar = tl.load(a_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)

    b_ptrs = rhs_ptr + offs_n * stride_bk + rk * stride_bk
    b = tl.load(b_ptrs, mask=offs_n < k, other=0.0)

    acc = tl.dot(ar, b, allow_tf32=False)

    acc_ptrs = output_ptr + offs_m[:, None] * stride_ok + offs_n[None, :] * stride_om
    tl.store(acc_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def _get_config():
    configs = [
        triton.Config({"BLOCK_M": 16}, num_stages=1, num_warps=8),
        triton.Config({"BLOCK_M": 16}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 16}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 16}, num_stages=8, num_warps=8),
        triton.Config({"BLOCK_M": 16}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 16}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 16}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 16}, num_stages=8, num_warps=4),
        triton.Config({"BLOCK_M": 16}, num_stages=1, num_warps=2),
        triton.Config({"BLOCK_M": 16}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_M": 16}, num_stages=4, num_warps=2),
        triton.Config({"BLOCK_M": 16}, num_stages=8, num_warps=2),
        triton.Config({"BLOCK_M": 16}, num_stages=1, num
