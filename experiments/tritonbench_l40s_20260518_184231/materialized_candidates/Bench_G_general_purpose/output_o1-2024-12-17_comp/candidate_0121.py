import triton
import triton.language as tl

def heuristics_for_tile_size(shape, max_tile_size=1024):
    n = shape[0]
    return min(n, max_tile_size)

def heuristics_for_num_warps(tile_size):
    if tile_size <= 256:
        return 1
    elif tile_size <= 512:
        return 2
    elif tile_size <= 1024:
        return 4
    return 8

class StridedBuffer:
    def __init__(self, base_ptr, shape, stride):
        self.base_ptr = base_ptr
        self.shape = shape
        self.stride = stride

@triton.jit
def relu_forward_kernel_rank_1(
    inp_ptr, out_ptr,
    n_elements,
    stride_in, stride_out,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(inp_ptr + offsets * stride_in, mask=mask, other=0.0)
    y = tl.where(x > 0.0, x, 0.0)
    tl.store(out_ptr + offsets * stride_out, y, mask=mask)

def relu_forward_wrapper_rank_1(inp: StridedBuffer, out: StridedBuffer):
    n = inp.shape[0]
    tile_size = heuristics_for_tile_size(inp.shape)
    num_warps = heuristics_for_num_warps(tile_size)
    grid = ((n + tile_size - 1) // tile_size,)
    relu_forward_kernel_rank_1[grid](
        inp_ptr=inp.base_ptr,
        out_ptr=out.base_ptr,
        n_elements=n,
        stride_in=inp.stride[0],
        stride_out=out.stride[0],
        BLOCK_SIZE=tile_size,
        num_warps=num_warps,
        num_stages=2
    )
