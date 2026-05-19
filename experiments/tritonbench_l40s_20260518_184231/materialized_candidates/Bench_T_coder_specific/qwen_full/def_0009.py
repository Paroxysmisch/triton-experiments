import torch
import triton
import triton.language as tl

@triton.jit
def _grid_sample_2d(
    I, X, Y, O, N, C, IH, IW, TH, TW, grid_d, grid_h, grid_w, padding_mode, mode,
    align_corners: tl.constexpr, ALLOW_SUBPIXEL: tl.constexpr, BLOCK: tl.constexpr,
):
    # Kernel logic here...

def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False) -> Tensor:
    # Wrapper function logic here...
