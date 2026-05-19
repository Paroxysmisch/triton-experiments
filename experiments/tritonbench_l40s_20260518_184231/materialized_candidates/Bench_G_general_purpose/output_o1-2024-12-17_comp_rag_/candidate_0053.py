import triton
import triton.language as tl
import torch

# ----------------------------------------------------------------------------
# Heuristic functions to determine tile size and the number of warps
# ----------------------------------------------------------------------------
def heuristics_for_tile_size(n_elems: int) -> int:
    # Simple heuristic: choose a tile size that is power of two and
    # doesn't exceed 1024 for demonstration. Adjust as needed.
    # This is a placeholder heuristic.
    tile_size = 1
    for size in [256, 512, 1024]:
        if n_elems >= size:
            tile_size = size
            break
    return tile_size

def heuristics_for_num_warps(tile_size: int) -> int:
    # Simple heuristic: pick 4 warps if tile_size >= 512, else 2 warps
    # This is a placeholder heuristic.
    if tile_size >= 512:
        return 4
    else:
        return 2

# ----------------------------------------------------------------------------
# Device function to check finiteness based on data type
# ----------------------------------------------------------------------------
@triton.jit
def _isfinite_func(x, is_fp64: tl.constexpr):
    # For demonstration, we branch on is_fp64. We rely on Python's
    # tl.isnan and tl.isinf for checking, which work for float32/float64
    if is_fp64:
        return ~(tl.isnan(x) | tl.isinf(x))
    else:
        return ~(tl.isnan(x) | tl.isinf(x))

# ----------------------------------------------------------------------------
# Triton Kernel for rank-1 tensor finiteness check
# ----------------------------------------------------------------------------
@triton.jit
def isfinite_func_kernel_rank_1(
    in_ptr,
    out_ptr,
    n_elems,    # total number of elements in the rank-1 tensor
    stride_in,  # stride for input
    stride_out, # stride for output
    BLOCK_SIZE: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
    is_fp64: tl.constexpr
):
    pid = tl.program_id(0)
    # Monolithic approach: single CTA processes entire tensor
    # or grid-stride-loop approach
    if one_tile_per_cta:
        # Each CTA handles a unique tile
        start_idx = pid * BLOCK_SIZE
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elems
        in_offsets = offsets * stride_in
        out_offsets = offsets * stride_out
        x = tl.load(in_ptr + in_offsets, mask=mask, other=0.0)
        result = _isfinite_func(x, is_fp64)
        tl.store(out_ptr + out_offsets, result, mask=mask)
    else:
        # Grid-stride loop
        block_start = pid * BLOCK_SIZE
        for elem_index in range(block_start, n_elems, BLOCK_SIZE * tl.num_programs(0)):
            offsets = elem_index + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elems
            in_offsets = offsets * stride_in
            out_offsets = offsets * stride_out
            x = tl.load(in_ptr + in_offsets, mask=mask, other=0.0)
            result = _isfinite_func(x, is_fp64)
            tl.store(out_ptr + out_offsets, result, mask=mask)

# ----------------------------------------------------------------------------
# Python Wrapper for rank-1 tensor finiteness check
# ----------------------------------------------------------------------------
def isfinite_func_wrapper_rank_1(input_tensor: torch.Tensor, output_tensor: torch.Tensor):
    assert input_tensor.dtype in (torch.float32, torch.float64), \
        "Only float32 or float64 supported."
    # Ensure output shape matches input shape
    assert input_tensor.shape == output_tensor.shape, \
        "Input and output must have the same shape for rank-1 kernel."
    assert len(input_tensor.shape) == 1, \
        "This kernel is specialized for rank-1."

    n_elems = input_tensor.shape[0]
    tile_size = heuristics_for_tile_size(n_elems)
    num_warps = heuristics_for_num_warps(tile_size)
    # Decide if we should process entire tensor with one tile or multiple CTAs
    one_tile_per_cta = (n_elems <= tile_size)

    grid = 1
    if not one_tile_per_cta:
        # We define number of CTAs so that each CTA can handle tile_size elements
        # in a grid-stride manner
        grid = (n_elems + tile_size - 1) // tile_size

    # Determine if data is fp64 or not
    is_fp64 = (input_tensor.dtype == torch.float64)
    # Launch kernel
    isfinite_func_kernel_rank_1[grid](
        in_ptr=input_tensor.data_ptr(),
        out_ptr=output_tensor.data_ptr(),
        n_elems=n_elems,
        stride_in=1,
        stride_out=1,
        BLOCK_SIZE=tile_size,
        one_tile_per_cta=one_tile_per_cta,
        is_fp64=is_fp64,
        num_warps=num_warps,
    )
