import torch
import triton
import triton.language as tl
from typing import Union, Tuple

@triton.jit
def sigmoid_adaptive_avg_pool2d_triton(x, W, H, C, r, s, y_ptr, x_row_stride, y_row_stride, y_col_stride, BLOCK: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    # Compute offsets for height and width
    h_offset = pid_h * BLOCK
    w_offset = pid_w * BLOCK
    # Compute offsets for height and width with rounding
    h_offset_rounded = (pid_h * BLOCK) + (BLOCK // 2)
    w_offset_rounded = (pid_w * BLOCK) + (BLOCK // 2)
    # Load input data with bounds checking
    mask_h = h_offset < H
    mask_w = w_offset < W
    mask = mask_h and mask_w
    x_offset = (h_offset * W + w_offset) * C
    x1 = tl.load(x + x_offset + 0 * C + 0, x_row_stride, eviction_policy='evict_last')
    x2 = tl.load(x + x_offset + 0 * C + 1, x_row_stride, eviction_policy='evict_last')
    x3 = tl.load(x + x_offset + 1 * C + 0, x_row_stride, eviction_policy='evict_last')
    x4 = tl.load(x + x_offset + 1 * C + 1, x_row_stride, eviction_policy='evict_last')
    # Apply sigmoid function
    y1 = 1 / (1 + tl.exp(-x1))
    y2 = 1 / (1 + tl.exp(-x2))
    y3 = 1 / (1 + tl.exp(-x3))
    y4 = 1 / (1 + tl.exp(-x4))
    # Store output data
    tl.store(y_ptr + (h_offset * W + w_offset) * C + 0 * C + 0, y1, y_row_stride)
    tl.store(y_ptr + (h_offset * W + w_offset) + 0 * C + 1, y2, y_row_stride)
    tl.store(y_ptr + (h_offset * W + w_offset) + 1 * C + 0, y3, y_row_stride)
    tl.store(y_ptr + (h_offset * W + w_offset) + 1 * C + 1, y4, y_row_stride)

def sigmoid_adaptive_avg_pool2d(input: torch.Tensor, output_size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    assert input.ndim == 4, "Input must be a 4D tensor"
    assert all((s > 0 for s in input.shape[1:])), "Input spatial dimensions must be greater than 0"
    assert all((s > 0 for s in (output_size if isinstance(output_size, tuple) else (output_size, output_size)))), "Output size must be greater than 0"

    N, C, H, W = input.shape
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    OH, OW = output_size
    output = torch.empty(N, C, OH, OW, dtype=input.dtype, device=input.device)
    # Create a grid of blocks for Triton kernel launch
    grid = lambda META: (triton.cdiv(H, META['BLOCK']), triton.cdiv(W, META['BLOCK']))
    # Launch Triton kernel
    sigmoid_adaptive_avg_pool2d_triton[grid](input, W, H, C, OH, OW, output, input.stride(1), output.stride(1), output.stride(2))
    return output
