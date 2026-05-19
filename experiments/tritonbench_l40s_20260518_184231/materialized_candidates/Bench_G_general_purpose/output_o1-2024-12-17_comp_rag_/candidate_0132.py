import triton
import triton.language as tl
import torch


def heuristics_for_tile_size(n: int, max_tile_size: int = 1024) -> int:
    """
    Computes an appropriate tile size for partitioning a 1D tensor.
    """
    if n <= max_tile_size:
        return n
    # Round up to the nearest divisor that doesn't exceed max_tile_size
    divisors = []
    for i in range(1, max_tile_size + 1):
        if max_tile_size % i == 0:
            divisors.append(max_tile_size // i)
    tile_size = min(divisors, key=lambda x: abs(x - (n ** 0.5)))
    return tile_size


def heuristics_for_num_warps(tile_size: int) -> int:
    """
    Determines number of warps based on tile size for performance.
    """
    if tile_size <= 64:
        return 1
    elif tile_size <= 128:
        return 2
    elif tile_size <= 256:
        return 4
    else:
        return 8


class StridedBuffer:
    """
    Represents a 1D buffer with a pointer, shape, and stride,
    supporting arbitrary (possibly negative) strides.
    """
    def __init__(self, tensor: torch.Tensor):
        self.data_ptr = tensor.data_ptr()
        self.shape = tensor.shape
        self.stride = tensor.stride()[0]
        self.dtype = tensor.dtype


@triton.jit
def relu_forward_kernel_rank_1(
    in_ptr, out_ptr,
    n, stride_in, stride_out,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(in_ptr + offsets * stride_in, mask=mask)
    y = tl.where(x > 0, x, 0.0)
    tl.store(out_ptr + offsets * stride_out, y, mask=mask)


def relu_forward_wrapper_rank_1(input_tensor: torch.Tensor, output_tensor: torch.Tensor):
    n = input_tensor.shape[0]
    tile_size = heuristics_for_tile_size(n)
    num_warps = heuristics_for_num_warps(tile_size)

    grid = lambda meta: ( (n + tile_size - 1) // tile_size, )
    relu_forward_kernel_rank_1[grid](
        in_ptr=input_tensor.data_ptr(),
        out_ptr=output_tensor.data_ptr(),
        n=n,
        stride_in=input_tensor.stride()[0],
        stride_out=output_tensor.stride()[0],
        BLOCK_SIZE=tile_size,
        num_warps=num_warps,
        num_stages=2
    )
