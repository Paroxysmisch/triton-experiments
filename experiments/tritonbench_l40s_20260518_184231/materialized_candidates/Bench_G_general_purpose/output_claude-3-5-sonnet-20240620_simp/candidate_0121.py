import triton
import triton.language as tl

@triton.jit
def relu_forward_kernel_rank_1(
    x_ptr, output_ptr,
    x_stride, output_stride,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets * x_stride, mask=mask)
    output = tl.maximum(x, 0)
    tl.store(output_ptr + offsets * output_stride, output, mask=mask)

class StridedBuffer:
    def __init__(self, ptr, shape, strides):
        self.ptr = ptr
        self.shape = shape
        self.strides = strides

def heuristics_for_tile_size(n_elements):
    if n_elements < 4096:
        return 128
    elif n_elements < 8192:
        return 256
    else:
        return 1024

def heuristics_for_num_warps(tile_size):
    if tile_size < 256:
        return 4
    elif tile_size < 1024:
        return 8
    else:
        return 16

def relu_forward_wrapper_rank_1(x: StridedBuffer, output: StridedBuffer):
    assert x.shape == output.shape, "Input and output shapes must match"
    n_elements = x.shape[0]

    tile_size = heuristics_for_tile_size(n_elements)
    num_warps = heuristics_for_num_warps(tile_size)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    relu_forward_kernel_rank_1[grid](
        x.ptr, output.ptr,
        x.strides[0], output.strides[0],
        n_elements,
        BLOCK_SIZE=tile_size,
        num_warps=num_warps
    )

# Example usage
import torch

def relu_forward(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    x_buffer = StridedBuffer(x.data_ptr(), x.shape, x.stride())
    output_buffer = StridedBuffer(output.data_ptr(), output.shape, output.stride())
    relu_forward_wrapper_rank_1(x_buffer, output_buffer)
    return output

# Test the function
if __name__ == "__main__":
    x = torch.randn(1000000, device='cuda')
    result = relu_forward(x)
    print("ReLU applied successfully")
    print("Input shape:", x.shape)
    print("Output shape:", result.shape)
    print("Sample input:", x[:5])
    print("Sample output:", result[:5])
