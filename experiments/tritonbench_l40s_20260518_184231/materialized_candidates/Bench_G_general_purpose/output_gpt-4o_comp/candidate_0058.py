import triton
import triton.language as tl

@triton.jit
def isfinite_func_kernel_rank_1(
    in_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr, 
    one_tile_per_cta: tl.constexpr
):
    # Calculate the start index for this CTA
    pid = tl.program_id(0)
    if one_tile_per_cta:
        start = pid * BLOCK_SIZE
        stride = 1
    else:
        start = pid
        stride = tl.num_programs(0)

    # Create a block pointer to the input and output tensors
    in0 = tl.load(in_ptr + start, mask=start < n_elements, other=0.0)
    out = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    # Check finiteness
    for i in range(0, BLOCK_SIZE):
        if start + i < n_elements:
            if tl.libdevice.isfinite(in0[i]):
                out[i] = 1
            else:
                out[i] = 0

    # Store results back to the output tensor
    tl.store(out_ptr + start, out, mask=start < n_elements)

def heuristics_for_tile_size(n_elements):
    # Simple heuristic: use a block size of 1024 or the number of elements, whichever is smaller
    return min(1024, n_elements)

def heuristics_for_num_warps(n_elements):
    # Simple heuristic: use 4 warps per block
    return 4

def isfinite_func_wrapper_rank_1(input_tensor, output_tensor):
    # Ensure input and output tensors have the same shape
    assert input_tensor.shape == output_tensor.shape

    # Determine the number of elements
    n_elements = input_tensor.numel()

    # Get heuristics for tile size and number of warps
    tile_size = heuristics_for_tile_size(n_elements)
    num_warps = heuristics_for_num_warps(n_elements)

    # Calculate the number of CTAs needed
    num_ctas = (n_elements + tile_size - 1) // tile_size

    # Determine if we should use one tile per CTA
    one_tile_per_cta = (num_ctas <= 1)

    # Set up the grid for execution
    grid = (num_ctas,)

    # Launch the Triton kernel
    isfinite_func_kernel_rank_1[grid](
        input_tensor, output_tensor, n_elements,
        BLOCK_SIZE=tile_size,
        one_tile_per_cta=one_tile_per_cta
    )

# Example usage:
# Assuming input_tensor and output_tensor are already allocated Triton tensors
# isfinite_func_wrapper_rank_1(input_tensor, output_tensor)
