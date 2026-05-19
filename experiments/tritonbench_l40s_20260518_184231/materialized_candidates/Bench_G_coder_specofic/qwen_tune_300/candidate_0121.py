import torch
import triton
import triton.language as tl
from flag_gems.utils.shape_utils import volume

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=8, num_warps=8),
        triton.Config({}, num_stages=1),
        triton.Config({}, num_stages=2),
        triton.Config({}, num_stages=4),
        triton.Config({}, num_stages=8),
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["tile_size"],
)
@triton.jit
def relu_forward_kernel_rank_1(
    input_pointer,
    output_pointer,
    size,
    tile_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    if pid * tile_size >= size:
        return

    tile_start = pid * tile_size
    tile_end = min((pid + 1) * tile_size, size)

    block_start = tile_start + tl.arange(0, BLOCK_SIZE)
    block_mask = block_start < tile_end

    input = tl.load(
        input_pointer + block_start, mask=block_mask, other=float(0)
    ).to(tl.float32)

    output = tl.where(input > 0, input, float(0))

    tl.store(
        output_pointer + block_start, output, mask=block_mask
    )


def heuristics_for_tile_size(size, config):
    max_tile_size = config["max_tile_size"]
    tile_size = min(volume(size), max_tile_size)
    return tile_size


def heuristics_for_num_warps(config):
    tile_size = config["tile_size"]
    num_warps = 8 if tile_size >= 2048 else 4
    num_warps = 2 if tile_size < 512 else num_warps
    return num_warps


class StridedBuffer:
    def __init__(self, tensor, stride=None):
        self.tensor = tensor
        if stride is None:
            self.stride = tensor.stride()
        else:
            self.stride = stride

    def offset(self, offset):
        return StridedBuffer(self.tensor[offset:], self.stride)

    def as_strided(self, size, stride):
        return StridedBuffer(self.tensor, stride=stride).offset(
            volume(size)
        )

    @property
    def device(self):
        return self.tensor.device

    def pin_memory(self):
        return StridedBuffer(self.tensor.pin_memory())

    @staticmethod
    def from_strided_buffer(strided_buffer, tensor):
        return StridedBuffer(tensor, stride=strided_buffer.stride)


def relu_forward_wrapper_rank_1(input, output):
    input_strided_buffer = StridedBuffer(input)
    output_strided_buffer = StridedBuffer(output)

    size = input.numel()
    tile_size = 2048

    grid = (triton.cdiv(size, tile_size),)

    BLOCK_SIZE = 128

    with torch.cuda.device(input.device):
        relu_forward_kernel_rank_1[grid](
            input_strided_buffer,
            output_strided_buffer,
            size,
            tile_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
