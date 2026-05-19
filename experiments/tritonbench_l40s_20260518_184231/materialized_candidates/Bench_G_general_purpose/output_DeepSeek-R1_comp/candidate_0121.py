import triton
import triton.language as tl

# Calculate optimal tile size based on input dimensions and max allowed size
def heuristics_for_tile_size(n_elements, max_tile_size=1024):
    tile_size = 1
    while tile_size < max_tile_size and tile_size * 2 <= n_elements:
        tile_size *= 2
    return min(tile_size, max_tile_size)

# Determine number of warps based on tile size (smaller tiles use fewer warps)
def heuristics_for_num_warps(tile_size):
    return max(tile_size // 256 * 4, 1)

# Core ReLU kernel for 1D tensors with strided memory access
@triton.jit
def relu_forward_kernel_rank_1(
    input_ptr, output_ptr,
    n_elements,
    input_stride, output_stride,
    element_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Calculate byte offsets using element size and strides
    input_byte_offsets = offsets * input_stride * element_size
    output_byte_offsets = offsets * output_stride * element_size

    # Load, compute ReLU, and store with boundary checking
    x = tl.load(input_ptr + input_byte_offsets, mask=mask)
    y = tl.where(x > 0, x, 0)
    tl.store(output_ptr + output_byte_offsets, y, mask=mask)

# Kernel launcher and parameter setup
def relu_forward_wrapper_rank_1(input_tensor, output_tensor):
    assert input_tensor.is_cuda and output_tensor.is_cuda
    assert input_tensor.dim() == 1 and output_tensor.dim() == 1
    
    n_elements = input_tensor.numel()
    assert n_elements == output_tensor.numel()

    # Calculate execution parameters
    tile_size = heuristics_for_tile_size(n_elements)
    num_warps = heuristics_for_num_warps(tile_size)
    grid = ((n_elements + tile_size - 1) // tile_size,)

    # Get memory access parameters
    element_size = input_tensor.element_size()
    input_stride = input_tensor.stride(0)
    output_stride = output_tensor.stride(0)

    # Launch kernel
    relu_forward_kernel_rank_1[grid](
        input_tensor.data_ptr(),
        output_tensor.data_ptr(),
        n_elements,
        input_stride,
        output_stride,
        element_size,
        BLOCK_SIZE=tile_size,
        num_warps=num_warps
    )
