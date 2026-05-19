import triton
import triton.language as tl

from .utils import heuristics_for_num_warps, heuristics_for_tile_size
from .single_kernel_utils import get_kernel_meta, rank_zero_to_taped
from .math_function_utils import _isfinited, _finitef

def isfinite_func_wrapper_rank_1(
    in0: torch.Tensor, out0: torch.Tensor, one_tile_per_cta: bool = False
):
    in0, out0 = rank_zero_to_taped([in0, out0])

    assert_same_dtype_as_first(in0, out0)
    in0 = in0.contiguous()
    out0 = out0.contiguous()
    assert in0.shape == out0.shape, f"in0.shape = {in0.shape}, out0.shape = {out0.shape}"

    N = in0.numel()

    tile_size = heuristics_for_tile_size(N)
    num_warps = heuristics_for_num_warps(tile_size)
    num_ctas = triton.cdiv(N, tile_size)

    grid = (num_ctas,)

    single_kernel_meta = get_kernel_meta()

    with torch.cuda._DeviceGuard(in0.device.index):
        isfinite_func_kernel_rank_1[grid](
            in0,
            out0,
            tile_size=tile_size,
            num_warps=num_warps,
            one_tile_per_cta=one_tile_per_cta,
            num_ctas=num_ctas,
            meta=single_kernel_meta,
        )
    return out0

@triton.jit
def isfinite_func_kernel_rank_1(
    in0_ptr, out0_ptr, tile_size: tl.constexpr, num_warps: tl.constexpr, one_tile_per_cta: tl.constexpr, num_ctas: tl.constexpr
):

    in0_dtype_name = tl.dtype_name(in0_ptr.dtype)

    pid = tl.program_id(0)

    tile_idx = pid if one_tile_per_cta else pid // num_warps
    offset = tile_idx * tile_size + tl.arange(0, tile_size)

    mask = offset < num_ctas * tile_size
    in0_ptrs = in0_ptr + offset
    out0_ptrs = out0_ptr + offset

    in0 = tl.load(in0_ptrs, mask=mask, other=float("nan"))

    out0 = in0

    if in0_dtype_name == "torch.float64":
        out0 = _isfinited(in0)
    else:
        out0 = _finitef(in0)

    tl.store(out0_ptrs, out0.to(tl.int64), mask=mask)
