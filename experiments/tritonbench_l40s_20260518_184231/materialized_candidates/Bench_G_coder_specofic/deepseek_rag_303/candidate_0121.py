import torch
import triton
import triton.language as tl
from flag_gems.auto_device import TritonAutotuner

class StridedBuffer:

    def __init__(self, data: torch.Tensor, view_size, stride):
        self.data = data
        self.view_size = view_size
        self.stride = stride

    def get(self, i):
        return self.data[i * self.stride:i * self.stride + self.view_size]

def heuristics_for_tile_size(n_elements, MAX_TILE_SIZE):
    n_tiles = (n_elements + MAX_TILE_SIZE - 1) // MAX_TILE_SIZE
    tile_size = min(n_elements, MAX_TILE_SIZE)
    return tile_size, n_tiles

def heuristics_for_num_warps(tile_size):
    if tile_size <= 64:
        # Small load/store is best done in a single warp, to avoid memory bank conflicts.
        num_warps = 1
    elif tile_size <= 128:
        # Best done in dual warps, to avoid L2 bank conflicts.
        num_warps = 2
    elif tile_size <= 2048:
        # Best done in two different warps to avoid warp scheduler conflicts.
        num_warps = 4
    else:
        # For very wide tiles, no need to use as many warps as there are elements, to avoid partial core utilization
        num_warps = 8

    return num_warps

@TritonAutotuner.register("relu_forward")
@triton.jit
def relu_forward_kernel_rank_1(
        x_ptr,
        y_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.where(x >= 0, x, 0)
    tl.store(y_ptr + offsets, y, mask=mask)

def relu_forward_wrapper_rank_1(x, out):
    n_elements = x.numel()

    MAX_TILE_SIZE = triton.next_power_of_2(n_elements)
    tile_size, n_tiles = heuristics_for_tile_size(n_elements, MAX_TILE_SIZE)
    num_warps = heuristics_for_num_warps(tile_size)

    grid = (n_tiles,)

    x_stride = int(x.stride(0))
    out_stride = int(out.stride(0))

    with torch.cuda.device(x.device.index):
        relu_forward_kernel_rank_1[grid](
            x,
            out,
            n_elements,
            BLOCK_SIZE=tile_size,
            num_warps=num_warps,
            x_stride=x_stride,
            out_stride=out_stride,
        )

def relu(
    x: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(x)
    relu_forward_wrapper_rank_1(x, out)
    return out
