import triton
import triton.language as tl

@triton.jit
def fused_hstack_div_kernel(
    x_ptr, divisor_ptr, y_ptr,
    n_tensors, tensor_shapes, stack_shape, divisor_shape,
    stride_x, stride_y, stride_d,
    block_size: tl.constexpr):
    
    # Calculate the global index
    row = tl.program_id(0)
    col = tl.program_id(1)

    # Load elements from input tensors
    x_val = tl.zeros_like(x_ptr[0])
    for i in range(n_tensors):
        if row < tensor_shapes[i]:
            idx = row * stride_x + col
            x_val += x_ptr[idx]

    # Load divisor
    divisor_val = divisor_ptr[col]

    # Perform division with optional rounding
    if rounding_mode == 'trunc':
        y_val = tl.floor(x_val / divisor_val)
    elif rounding_mode == 'floor':
        y_val = tl.floor(x_val / divisor_val)
    else:
        y_val = x_val / divisor_val

    # Store the result
    if row < stack_shape[0]:
        idx = row * stride_y + col
        y_ptr[idx] = y_val

@triton.autotune(
    configs=[
        triton.Config({'block_size': 256}, num_stages=2, num_warps=4),
        triton.Config({'block_size': 512}, num_stages=2, num_warps=8),
        triton.Config({'block_size': 1024}, num_stages=2, num_warps=16),
    ],
    key=['n_tensors', 'tensor_shapes', 'stack_shape', 'divisor_shape']
)
def fused_hstack_div(x, divisor, y, n_tensors, tensor_shapes, stack_shape, divisor_shape, rounding_mode='none'):
    n_blocks = (stack_shape[0] * stack_shape[1] - 1) // 256 + 1
    grid = (n_blocks, 1)
    block_size = 256
    fused_hstack_div_kernel[x.shape[0], block_size](x, divisor, y, n_tensors, tensor_shapes, stack_shape, divisor_shape, x.stride(0), y.stride(0), divisor.stride(0), block_size)
