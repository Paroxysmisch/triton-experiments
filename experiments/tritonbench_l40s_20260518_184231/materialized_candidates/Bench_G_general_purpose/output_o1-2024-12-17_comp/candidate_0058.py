import triton
import triton.language as tl


def heuristics_for_tile_size(shape):
    return 256


def heuristics_for_num_warps(shape):
    return 4


@triton.jit
def _isfinited(x):
    return 1 - tl.isnan(x + 0.0) - tl.isinf(x)


@triton.jit
def _finitef(x):
    return 1 - tl.isnan(x + 0.0) - tl.isinf(x)


@triton.jit
def isfinite_func_kernel_rank_1(
    in_ptr, out_ptr, n_elements, one_tile_per_cta,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    # Monolithic approach: each CTA handles a contiguous range of size BLOCK_SIZE
    # Grid-stride approach: each thread block processes multiple tiles in a loop
    start_offset = pid * BLOCK_SIZE if one_tile_per_cta else pid * BLOCK_SIZE * 1

    offsets = start_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    in_ptrs = in_ptr + offsets
    in_vals = tl.load(in_ptrs, mask=mask, other=0.0)

    # Determine if we might need fp64 checking
    # Here we assume all fp32 for demonstration. Adjust logic as needed.
    is_fp64 = False
    out_vals = tl.where(is_fp64, _isfinited(in_vals), _finitef(in_vals))

    # Store results
    out_ptrs = out_ptr + offsets
    tl.store(out_ptrs, out_vals, mask=mask)

    # If we aren't in a one_tile_per_cta mode, loop over remaining tiles
    if not one_tile_per_cta:
        tile_step = BLOCK_SIZE * tl.num_programs(0)
        remain = start_offset + tile_step
        while remain < n_elements:
            offsets = remain + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            in_ptrs = in_ptr + offsets
            in_vals = tl.load(in_ptrs, mask=mask, other=0.0)
            out_vals = tl.where(is_fp64, _isfinited(in_vals), _finitef(in_vals))
            out_ptrs = out_ptr + offsets
            tl.store(out_ptrs, out_vals, mask=mask)
            remain += tile_step


def isfinite_func_wrapper_rank_1(input_tensor, output_tensor):
    if input_tensor.shape != output_tensor.shape:
        raise ValueError("Shapes of input and output tensors must match.")

    n_elements = input_tensor.numel()
    tile_size = heuristics_for_tile_size(input_tensor.shape)
    num_warps = heuristics_for_num_warps(input_tensor.shape)

    # Decide if each CTA handles exactly one tile (True) or multiple tiles (False)
    one_tile_per_cta = (n_elements <= tile_size * 65535)
    if one_tile_per_cta:
        num_ctas = (n_elements + tile_size - 1) // tile_size
    else:
        # Assign multiple tiles per CTA if needed
        num_ctas = 65535

    grid = (num_ctas,)

    isfinite_func_kernel_rank_1[grid](
        input_tensor.data_ptr(),
        output_tensor.data_ptr(),
        n_elements,
        one_tile_per_cta,
        BLOCK_SIZE=tile_size,
        num_warps=num_warps,
    )
