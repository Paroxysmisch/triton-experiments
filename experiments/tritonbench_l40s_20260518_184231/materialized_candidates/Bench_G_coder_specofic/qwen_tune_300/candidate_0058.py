import torch
import triton
import triton.language as tl
from flag_gems.utils.libutils import get_device_compute_capability

def heuristics_for_tile_size(args):
    if args["num_dims"] == 1:
        if args["elem_sz"] == 8:
            return 1024
        else:
            return 2048
    else:
        return heuristics_for_tile_size(args)

def heuristics_for_num_warps(args):
    if args["num_dims"] == 1:
        return 4
    else:
        return heuristics_for_num_warps(args)

@triton.jit
def isfinite_func_kernel_rank_1(
    in0,
    out0,
    n_elements,
    one_tile_per_cta: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    # Compute boundary
    block_idx = tl.program_id(0)
    if one_tile_per_cta:
        assert n_elements <= TILE_SIZE
        elements_remaining = n_elements
        start_idx = block_idx * TILE_SIZE
        offsets = start_idx + tl.arange(0, TILE_SIZE)
    else:
        elements_remaining = n_elements
        offsets = block_idx * TILE_SIZE + tl.arange(0, TILE_SIZE)
    mask = offsets < n_elements

    # Load in0
    in0_ptr = in0 + offsets
    in0_val = tl.load(in0_ptr, mask=mask, other=0.0)

    # Call the function
    out0_val = isfinite_func(in0_val, offsets, mask)

    # Store out0
    out0_ptr = out0 + offsets
    tl.store(out0_ptr, out0_val, mask=mask)

def isfinite_func_wrapper_rank_1(in0, *, dtype=None):
    assert in0.is_contiguous()
    assert in0.dim() == 1

    n_elements = in0.numel()
    if dtype is None:
        dtype = in0.dtype
    out0 = torch.empty_like(in0, dtype=dtype, device=in0.device)

    grid = lambda meta: (triton.cdiv(n_elements, meta["TILE_SIZE"]),)
    heuristics = [
        ("TILE_SIZE", heuristics_for_tile_size),
        ("num_warps", heuristics_for_num_warps),
    ]
    is_hip = get_device_compute_capability(in0.device) is not None
    isfinite_func_kernel_rank_1[grid](in0, out0, n_elements, heuristics=heuristics)
    return out0

def isfinite_func(x, offsets, mask):
    if x.dtype == torch.float64:
        return _isfinited(x)
    else:
        return _finitef(x.to(torch.float32))

def _isfinited(x):
    return x.isfinite()

def _finitef(x):
    return x.isfinite()
