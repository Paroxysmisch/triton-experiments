import torch
import triton
import triton.language as tl
from triton import RuntimeError
from triton.runtime.jit import _overload_manager


def matmul(input, other, *, out=None):
    if not isinstance(input, torch.Tensor) or not isinstance(other, torch.Tensor):
        raise TypeError("input and other must be torch.Tensor")

    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("out must be None or a torch.Tensor")

    if out is not None and out.ndim != max(input.ndim, other.ndim):
        raise RuntimeError(
            f"out must have the same number of dimensions as input ({input.ndim}) or other ({other.ndim})"
        )

    if out is not None and out.shape != input.shape:
        raise RuntimeError(
            f"shape of out must be the same as input ({input.shape})"
        )

    if input.ndim == 1 and other.ndim == 1:
        return _matmul1d1d(input, other, out)

    if input.ndim == 2 and other.ndim == 2:
        return _matmul2d2d(input, other, out)

    if input.ndim in (1, 2) and other.ndim == 1:
        return _matmul1d2d(input, other, out)

    if input.ndim == 2 and other.ndim == 1:
        return _matmul2d1d(input, other, out)

    if input.ndim == 2 and other.ndim == 2:
        return _matmul2d2d(input, other, out)

    if input.ndim > 2 or other.ndim > 2:
        return _matmul_batched(input, other, out)


@triton.jit
def _matmul1d1d(a, b, out):
    idx = tl.arange(0, 1)
    dot_product = tl.sum(a[idx] * b[idx])
    tl.store(out, dot_product)


@triton.jit
def _matmul2d2d(a, b, out):
    M, K = a.shape
    K, N = b.shape
    pid = tl.program_id(0)
    prune_config = tl.PruneConfig(
        iter_vars=[tl.num_programs(0)],
        prune_depth=2,
        num_warps=1,
        num_stages=1,
    )
    a = tl.make_block_ptr(a, (M, K), (tl.num_programs(1) * N, K), (pid * N, 0), (1, 0), (N, 1))
    b = tl.make_block_ptr(b, (K, N), (K, N), (0, pid * 1), (1, 0), (1, N))
    c = tl.zeros([1, 1], dtype=tl.float32)
    acc = tl.zeros([1, 1], dtype=tl.float32)
    for _ in range(0, 64):
        offs_m = tl.arange(0, 1)
        offs_n = tl.arange(0, 1)
        a_ptrs = tl.advance(a, offs_m[:, None] * K + offs_n[None, :] * 1)
        b_ptrs = tl.advance(b, offs_m[:, None] * 1 + offs_n[None, :] * N)
        a_val = tl.load(a_ptrs, boundary_check=(0, 1))
        b_val = tl.load(b_ptrs, boundary_check=(0, 1))
        c = tl.dot(a_val, b_val, acc=c)
    c = tl.sum(c, axis=1)
    c = tl.sum(c, axis=0)
    tl.store(out, c)


@triton.jit
def _matmul1d2d(a, b, out):
    M, K = b.shape
    a = a.to(tl.float32)
    b = tl.make_block_ptr(b, (K, M), (M, 1), (0, 0), (1, 0), (1, M))
    c = tl.zeros([1, M], dtype=tl.float32)
    acc = tl.zeros([1, M], dtype=tl.float32)
    for _ in range(0, 64):
        offs_n = tl.arange(0, 1)
        b_ptrs = tl.advance(b, offs_n[None, :] * M)
        b_val = tl.load(b_ptrs, boundary_check=(0, 1))
        c = tl.dot(a, b_val, acc=c)
    c = tl.sum(c, axis=1)
    c = tl.sum(c, axis=0)
    tl.store(out, c)


@triton.jit
def _matmul2d1d(a, b, out):
    K, N = a.shape
    a = tl.make_block_ptr(a, (K, N), (N, 1), (0, 0), (1, 0), (1, N))
    b = b.to(tl.float32)
    c = tl.zeros([N, 1], dtype=tl.float32)
    acc = tl.zeros([N, 1], dtype=tl.float32)
    for _ in range(0, 64):
        offs_n = tl.arange(0, 1)
        a_ptrs = tl.advance(a, offs_n[None, :] * N)
        a_val = tl.load(a_ptrs, boundary_check=(0, 1))
        c = tl.dot(a_val, b, acc=c)
    c = tl.sum(c, axis=0)
    c = tl.sum(c, axis=0)
    tl.store(out, c)


@triton.jit
def _matmul_batched(a, b, out):
    a_b, a_m, a_k = a.shape
    b_k, b_n = b.shape
    c = torch.empty((a_b, a_m, b_n), dtype=a.dtype, device=a.device)
    prune_config = tl.PruneConfig(
        iter_vars=[tl.num_programs(0)],
        prune_depth=2,
        num_warps=1,
        num_stages=1,
    )
    for i in range(a_b):
        a_ptrs = tl.make_block_ptr(a, (a_m, a_k), (b_n * a_k, a_k), (i * a_m * a_k, 0), (a_k, 1), (a_m * a_k, a_k))
        b_ptrs = tl.make_block_ptr(b, (a_k, b_n), (b_n, 1), (0, 0), (1, 0), (b_n, b_n))
        c_ptrs = tl.make_block_ptr(c, (a_m, b_n), (b_n, 1), (i * a_m * b_n, 0), (1, 0), (a_m * b_n, b_n))
        _matmul2d2d(a_ptrs, b_ptrs, c_ptrs, prune_config=prune_config)
    tl.debug_barrier()
    if out is not None:
        out.
