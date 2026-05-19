import torch
import triton
import triton.language as tl

@triton.jit
def heuristics_for_tile_size(args):
    max_tile_size = args["MAX_TILE_SIZE"]
    size = args["SIZE"]
    num_stages = tl.cdiv(size, max_tile_size)
    return num_stages

@triton.jit
def heuristics_for_num_warps(args):
    tile_size = args["TILE_SIZE"]
    if tile_size < 1024:
        num_warps = 4
    elif tile_size < 2048:
        num_warps = 8
    else:
        num_warps = 16
    return num_warps

class StridedBuffer:
    def __init__(self, data, offset, stride, size):
        self.data = data
        self.offset = offset
        self.stride = stride
        self.size = size

    def __getitem__(self, item):
        if isinstance(item, slice):
            start = item.start if item.start is not None else 0
            stop = item.stop if item.stop is not None else 0
            step = item.step if item.step is not None else 1
            size = (stop - start + step - 1) // step
            offset = self.offset + start * self.stride
            stride = self.stride * step
            return StridedBuffer(self.data, offset, stride, size)
        else:
            raise ValueError("StridedBuffer indices can only be slices")

    def to_tensor(self):
        if self.data.ndim == 0:
            return self.data.view(torch.float32).repeat(self.size)
        else:
            return self.data[self.offset : self.offset + self.stride * self.size : self.stride]

@triton.jit
def relu_forward_kernel_rank_1(
    x_ptr,
    y_ptr,
    x_stride_0,
    x_stride_1,
    y_stride_0,
    y_stride_1,
    x_size_0,
    x_size_1,
    y_size_0,
    y_size_1,
    BLOCK_SIZE_0: tl.constexpr,
    BLOCK_SIZE_1: tl.constexpr,
):
    program_id_0 = tl.program_id(axis=0)
    program_id_1 = tl.program_id(axis=1)
    x_stride_0 = x_stride_0 // 8
    x_stride_1 = x_stride_1 // 8
    y_stride_0 = y_stride_0 // 8
    y_stride_1 = y_stride_1 // 8
    x_size_0 = x_size_0 // BLOCK_SIZE_0
    x_size_1 = x_size_1 // BLOCK_SIZE_1
    y_size_0 = y_size_0 // BLOCK_SIZE_0
    y_size_1 = y_size_1 // BLOCK_SIZE_1
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(x_size_0, x_size_1),
        strides=(x_stride_0, x_stride_1),
        offsets=(program_id_0 * BLOCK_SIZE_0, program_id_1 * BLOCK_SIZE_1),
        block_shape=(BLOCK_SIZE_0, BLOCK_SIZE_1),
        order=(1, 0),
    )
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(y_size_0, y_size_1),
        strides=(y_stride_0, y_stride_1),
        offsets=(program_id_0 * BLOCK_SIZE_0, program_id_1 * BLOCK_SIZE_1),
        block_shape=(BLOCK_SIZE_0, BLOCK_SIZE_1),
        order=(1, 0),
    )
    x = tl.load(x_block_ptr)
    x_neg = x < 0
    y = tl.where(x_neg, 0, x)
    tl.store(y_block_ptr, y)

def relu_forward_wrapper_rank_1(x, y):
    x_strides = list(x.stride())
    y_strides = list(y.stride())
    x_neg_strides = [-i for i in x_strides]
    x_neg = StridedBuffer(x, 0, -i, x.size(0))
    y_neg = StridedBuffer(y, 0, -i, y.size(0))
    grid = lambda META: (
        triton.cdiv(x.size(0), META["BLOCK_SIZE_0"]),
        triton.cdiv(x.size(1), META["BLOCK_SIZE_1"]),
    )
    relu_forward_kernel_rank_1[grid](
        x_neg,
        y_neg,
        *x_strides,
        *y_strides,
        *x.size(),
        *y.size(),
    )
