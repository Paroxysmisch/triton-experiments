import torch
import triton
import triton.language as tl
from vllm.platforms import current_platform

@triton.jit
def isfinite_func_kernel_rank_1(
    in0,
    out0,
    one_tile_per_cta,
    tile_size: tl.constexpr,
    # heuristics
    num_warps: tl.constexpr,
):
    # kernel logic
    pid = tl.program_id(axis=0)
    if one_tile_per_cta:
        tid = pid
        offset = tid * tile_size
        in0 = tl.make_block_ptr(in0, (offset,), (tile_size,), (1,), (0,))
        out0 = tl.make_block_ptr(out0, (offset,), (tile_size,), (1,), (0,))
        tile = tl.arange(0, tile_size)
    else:
        tile = tl.arange(0, tile_size) + pid * tile_size
        in0 = tl.make_block_ptr(in0, (0,), (tile_size,), (1,), (pid * tile_size,))
        out0 = tl.make_block_ptr(out0, (0,), (tile_size,), (1,), (pid * tile_size,))
    in0 = tl.load(in0, boundary_check=(0,))
    in0 = tl.where(
        tl.math.isfinited(in0) if in0.dtype.element_ty == tl.float64 else tl.math.finitef(in0.to(tl.float32))
    )
    tl.store(out0, in0, boundary_check=(0,))

def heuristics_for_tile_size(args):
    if args["one_tile_per_cta"]:
        return 1
    tile_size = args["tile_size"]
    if tile_size < 1024:
        return 16
    if tile_size < 2048:
        return 32
    return 64

def heuristics_for_num_warps(args):
    tile_size = args["tile_size"]
    if tile_size <= 16:
        return 4
    if tile_size <= 32:
        return 8
    if tile_size <= 64:
        return 16
    if tile_size <= 128:
        return 24
    if tile_size <= 256:
        return 32
    return 48

def isfinite_func_wrapper_rank_1(
    in0,
    *,
    tile_size=None,
    one_tile_per_cta=None,
):
    in0_shape_str = ".".join(str(x) for x in in0.shape)
    in0 = in0.contiguous()
    out0 = torch.empty_like(in0, dtype=torch.bool)
    assert in0.is_contiguous()
    in0_stride0 = in0.stride(0)
    in0_stride1 = in0.stride(1) if len(in0.stride()) > 1 else 1
    out0_stride0 = out0.stride(0)
    out0_stride1 = out0.stride(1) if len(out0.stride()) > 1 else 1

    if tile_size is None:
        tile_size = min(1024, triton.next_power_of_2(in0.shape[0]))
    if one_tile_per_cta is None:
        one_tile_per_cta = in0.numel() <= 65536
    num_ctas = triton.cdiv(in0.numel(), tile_size)
    num_ctas = min(num_ctas, current_platform.num_max_ctas())
    grid = (num_ctas,)
    num_warps = heuristics_for_num_warps({"tile_size": tile_size, "one_tile_per_cta": one_tile_per_cta})

    isfinite_func_kernel_rank_1[grid](
        in0,
        out0,
        one_tile_per_cta,
        tile_size,
        num_warps=num_warps,
        num_ctas=num_ctas,
    )
    return out0
