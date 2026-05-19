import triton
import triton.language as tl
import torch

def heuristics_for_tile_size(input_size, max_tile_size=1024):
    # Calculate appropriate tile size
    tile_size = min(max_tile_size, input_size)
    return tile_size

def heuristics_for_num_warps(tile_size):
    # Determine number of warps based on tile size
    if tile_size <= 64:
        return 1
    elif tile_size <= 128:
        return 2
    elif tile_size <= 256:
        return 4
    elif tile_size <= 512:
        return 8
    else:
        return 16

class StridedBuffer:
    def __init__(self, tensor):
        self.data = tensor
        self.stride = tensor.stride()
        self.shape = tensor.shape

@triton.jit
def relu_forward_kernel_rank_1(
    input_ptr, output_ptr, size, stride_input, stride_output, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    input_ptrs = input_ptr + offsets * stride_input
    output_ptrs = output_ptr + offsets * stride_output

    input_vals = tl.load(input_ptrs, mask=mask)
    output_vals = tl.where(input_vals > 0, input_vals, 0.0)
    tl.store(output_ptrs, output_vals, mask=mask)

def relu_forward_wrapper_rank_1(input_tensor, output_tensor):
    input_buffer = StridedBuffer(input_tensor)
    output_buffer = StridedBuffer(output_tensor)

    size = input_tensor.numel()
    tile_size = heuristics_for_tile_size(size)
    num_warps = heuristics_for_num_warps(tile_size)

    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    relu_forward_kernel_rank_1[grid](
        input_buffer.data, output_buffer.data, size,
        input_buffer.stride[0], output_buffer.stride[0],
        BLOCK_SIZE=tile_size, num_warps=num_warps
    )

# Example usage
input_tensor = torch.randn(1024, device='cuda')
output_tensor = torch.empty_like(input_tensor)
relu_forward_wrapper_rank_1(input_tensor, output_tensor)
print(output_tensor)
