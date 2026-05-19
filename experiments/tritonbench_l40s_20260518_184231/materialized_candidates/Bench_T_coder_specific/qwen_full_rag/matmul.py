import torch
import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@triton.jit
def _matmul_a(
    acc,
    a_ptr,
    b_ptr,
    idx_m,
    idx_n,
    idx_k,
    M,
    N,
    K,
    AS1,
    AS2,
    BS1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    # load a tile
    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = idx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = idx_k * BLOCK_K + tl.arange(0, BLOCK_K)
    a_tile = tl.load(
        a_ptr
        + offs_m[:, None] * AS1
        + offs_k[None, :] * AS2
        + tl.broadcast_to(offs_k[None, :], (BLOCK_M, BLOCK_K)),
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        other=0.0,
    )
    b_tile = tl.load(b_ptr + offs_k[:, None] * BS1 + offs_n[None, :] * BS2, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
    if EVEN_K:
        c_tile = tl.dot(a_tile, b_tile, allow_tf32=ALLOW_TF32)
    else:
        c_tile = tl.dot(a_tile.to(tl.float32), b_tile.to(tl.float32), allow_tf32=ALLOW_TF32)
    # rematerialize offsets to save registers
    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = idx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # handles write-back with reduction-add
    tl.atomic_add(acc + offs_m[:, None] * N + offs_n[None, :], c_tile)

@triton.jit
def _matmul_b(
    acc,
    a_ptr,
    b_ptr,
    idx_m,
    idx_n,
    idx_k,
    M,
    N,
    K,
    AS1,
    AS2,
    BS1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    # load a tile
    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = idx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = idx_k * BLOCK_K + tl.arange(0, BLOCK_K)
    a_tile = tl.load(
        a_ptr
        + offs_m[:, None] * AS1
        + offs_k[None, :] * AS2
        + tl.broadcast_to(offs_k[None, :], (BLOCK_M, BLOCK_K)),
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        other=0.0,
    )
    b_tile = tl.load(b_ptr + offs_k[:, None] * BS1 + offs_n[None, :] * BS2, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
    if EVEN_K:
        c_tile = tl.dot(a_tile, b_tile, allow_tf32=ALLOW_TF32)
    else:
        c_tile = tl.dot(a_tile.to(tl.float32), b_tile.to(tl.float32), allow_tf32=ALLOW_TF32)
    # rematerialize offsets to save registers
    offs_n = idx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # handles write-back with reduction-add
    tl.atomic_add(acc + offs_m[:, None] * N + offs_n[None, :], c_tile)

def matmul(x, y, *, out=None):
    device = x.device
    if out is not None:
        assert out.is_contiguous()
        assert out.dtype == x.dtype
        assert out.shape == (
            x.shape[0],
            y.shape[1],
        ), "Shape mismatch for output"
        assert out.device == device
    else:
        out = torch.empty((x.shape[0], y.shape[1]), device=device, dtype=x.dtype)
    assert x.is_contiguous()
    assert y.is_contiguous()

    assert x.dim() >= 2 and y.dim() >= 2, "Only dimensions >= 2 are supported"

    M = x.shape[-2]
    N = y.shape[-1]
    K = x.shape[-1]

    assert (
        x.shape[-1] == y.shape[-2]
    ), f"Size must match on all but last dimension, but got {x.shape} @ {y.shape}"

    broadcast_shape = list(x.shape[:-2]) + [M, N]
    out = out.reshape(broadcast_shape)
    batch_shape = out.shape[:-2]

    # We don't want any synchronization overhead for small matrices
    if M * N <= 512:
        _matmul_no_synchronization(x, y, out)
        return out

    grid_fn = partial(grid, features=["divisible_by_16"], solution_index=1)

    block_size = 128 if x.dtype in (torch.float16, torch.bfloat16) else 64
    _, num_warps, _ = get_configs()["config0"]
    descriptor = instance_descriptor(divisible_by_16=True)
    pgm = torch.compile(
        lambda _: _matmul_a(
            out,
            x,
            y,
            0,
            0,
            0,
            M,
            N,
            K,
            x.stride(-2),
            x.stride(-1),
            y.stride(-2),
            BLOCK_M=block_size,
            BLOCK_N=block_size,
            BLOCK_K=32,
            GROUP_SIZE_M=1,
            EVEN_K=True,
            ALLOW_TF32=False,
            num_stages=3,
            num_warps=num_warps,
        ),
        signature="*fp32,*fp32,*fp32",
        constants={},
        device=0,
        stream=None,
        _instance_descriptor=descriptor,
        _pre_hook=None,
        _post_hook=None,
        _init_to_zero=[],
        _grid=grid_fn,
        _kernel_name=None,
    ).to(device)
    triton_helpers.run_kernel(
        pgm,
        ((out,) + (x, y)),
        {},
        grid=(*batch_shape, 1),
        stream=None,
        init_to_zero=[],
        pre_hook=None,
        post_hook=None,
    )

    return out
