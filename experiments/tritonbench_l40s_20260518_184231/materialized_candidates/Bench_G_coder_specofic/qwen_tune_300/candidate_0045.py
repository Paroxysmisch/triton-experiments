import torch
import triton
import triton.language as tl
from flag_gems.utils.shape_utils import volume
from flag_gems.utils import heuristics
from flag_gems.ops.triton import pow as triton_pow

def cfggen():
    block_m = [1, 2, 4]
    block_n = [1024, 2048, 4096]
    warps = [4, 8, 16]
    configs = [
        triton.Config({"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n}, num_warps=w)
        for m in block_m
        for n in block_n
        for w in warps
    ]
    return configs

@triton.jit
def pow_func_scalar_tensor_kernel_rank_1(
    in0_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_M
    arange = tl.arange(0, BLOCK_SIZE_N)
    offsets = block_start + arange
    mask = offsets < n_elements
    in0 = tl.load(in0_ptr + offsets, mask=mask)
    out = tl.pow(in0, 2)
    tl.store(out_ptr + offsets, out, mask=mask)

def pow_func_scalar_tensor_wrapper_rank_1(
    a,
    *,
    dtype=None,
    out=None,
    where=True,
    casting="same_kind",
    order="K",
    subok=True,
    signature=None,
    extobj=None,
):
    if dtype is None:
        dtype = a.dtype
    if out is None:
        out = torch.empty_like(a, dtype=dtype)
    assert a.shape == out.shape
    n_elements = volume(a.shape)
    grid_fn = heuristics.heuristic_grid_1d
    kwargs = {
        "num_stages": 1,
        "num_warps": 4,
    }
    pow_func_scalar_tensor_kernel_rank_1[grid_fn](
        a,
        out,
        n_elements,
        **kwargs,
    )
    return out
