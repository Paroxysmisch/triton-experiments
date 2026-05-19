import logging
import functools
import torch
import triton
import triton.language as tl
from flag_gems.utils.shape_utils import volume
from flag_gems.utils import layout_helpers
from flag_gems.ops.triton.kernels import gemm


def has_typed_mem_fn(layout):
    return layout in (torch.strided, torch.sparse_csr, torch.sparse_bsr)


def is_eligible(shape, dtype, layout, device, pin_memory):
    return (shape and len(shape) > 2) or has_typed_mem_fn(layout) or device.type == "rocm" or dtype == torch.float16


@triton.jit
def matmul_fallback_kernel(
    a_ptr, b_ptr, c_ptr, n_ptr, k_ptr, lda, ldb, ldc, N, K, M, grid_offset, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(0) + grid_offset
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * lda + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * ldb + offs_n[None, :])
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * ldb
    c = accumulator.to(tl.float16)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n = tl.load(n_ptr)
    k = tl.load(k_ptr)
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    c_ptrs = c_ptr + ldc * offs_m[:, None] + offs_n[None, :]
    tl.store(c_ptrs, c, mask=mask)


def matmul_fallback(
    a,
    b,
    n=None,
    k=None,
    *,
    out=None,
    allow_tf32=True,
    out_dtype=None,
    out_layout=None,
    out_device=None,
    out_pin_memory=None,
):
    logging.debug("GEMS MATMUL")
    device = a.device
    if not all((x.is_contiguous() for x in (a, b))):
        a, b = a.contiguous(), b.contiguous()
    if a.device.type == "cuda" and b.device.type == "cuda":
        if a.layout == torch.sparse_csr and b.layout == torch.sparse_csr:
            return torch.sparse.mm(a, b.to(torch.sparse_csr))
        elif a.layout == torch.sparse_csr:
            return torch.sparse.mm(a, b)
        elif b.layout == torch.sparse_csr:
            return torch.sparse.mm(b, a).T
    if not is_eligible(a.shape, a.dtype, a.layout, device, a.pin_memory) or not is_eligible(
        b.shape, b.dtype, b.layout, device, b.pin_memory
    ):
        logging.debug("GEMS MATMUL FALLBACK")
        if a.layout == torch.sparse_csr and b.layout == torch.sparse_bsr:
            if a._nnz() == 0 or b._nnz() == 0:
                return torch.zeros(a.size(0), b.size(1), device=a.device, dtype=a.dtype)
            if a.size(-1) != b.size(-2):
                raise ValueError("Size mismatch CSRCOO @ BSRCOO")
            if a._sparse_dim() != 2 or b._sparse_dim() != 2:
                raise RuntimeError("Only 2D sparse matmuls are supported currently")
            if a.size(0) != b.size(0):
                raise ValueError("Shape mismatch on batch dimensions CSRCOO @ BSRCOO")

            _n = b.size(-1)
            _k = a.size(-1)

            b = b.coalesce()
            a_indices = a._indices()
            a_values = a._values()
            b_indices = b._indices()
            b_values = b._values()

            _batch_indices = a_indices[:-2]
            _matmul_indices = a_indices[-2:]
            _n_repeat_indices = list(
                itertools.chain.from_iterable(
                    itertools.repeat(i, b.size(0) // a.size(0)) for i in range(a.size(0))
                )
            )
            _n_repeat_indices = torch.tensor(_n_repeat_indices, dtype=torch.int32, device=a.device)
            _n_indices = torch.arange(b.size(-1), device=a.device, dtype=torch.int32)
            mask = _n_indices != _n_repeat_indices[:, None]
            _n_indices = _n_indices[None, :] + (_n_repeat_indices[:, None] + 1) * b.size(-1)
            n_repeat = torch.sum(mask.to(torch.int32), axis=0)
            n = torch.full((1, 1), n_repeat, dtype=torch.int32, device=a.device)

            _k_repeat_indices = list(
                itertools.chain.from_iterable(
                    itertools.repeat(i, b.size(0) // a.size(0)) for i in range(a.size(0))
                )
            )
            _k_repeat_indices = torch.tensor(_k_repeat_indices, dtype=torch.int32, device=a.device)
            k_repeat = torch.abs(_k - _k_repeat_indices)
            k = torch.min(k_repeat, axis=0)[0]

            output = torch.empty(a_indices.size(0) - 2, n, device=a.device, dtype=a.dtype)
            output._indices_(a_indices)
            output._values_(torch.zeros_like(a_values))
            matmul_fn = functools.partial(
                matmul_fallback_kernel,
                n_ptr=n,
                k_ptr=k,
                lda=a.stride(-2),
                ldb=b.stride(-2),
                ldc=output.stride(-2),
                M=a.size(-2),
                N=b.size(-1),
                K=a.size(-1),
            )
            grid = lambda META: (triton.cdiv(a.size(-2), META["BLOCK_SIZE_M"]) * triton.cdiv(b.size(-1), META["BLOCK_SIZE_N"]),)
            matmul_fn[grid]()
            return output.to_sparse_csr()
        else:
            return torch.matmul(a, b)
    else:
        return matmul_triton(a, b, out, allow_tf32, out_dtype, out_layout, out_device, out_pin_memory)
