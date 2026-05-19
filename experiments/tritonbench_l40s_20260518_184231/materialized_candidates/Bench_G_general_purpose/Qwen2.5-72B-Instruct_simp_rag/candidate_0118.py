import triton
import triton.language as tl
import torch

# Heuristics for determining tile sizes
def heuristics_for_tile_size(input_size):
    if input_size < 1024:
        return 32
    elif input_size < 4096:
        return 64
    else:
        return 128

# Heuristics for determining the number of warps
def heuristics_for_num_warps(tile_size):
    if tile_size <= 64:
        return 4
    else:
        return 8

# Custom class for handling tensors with arbitrary strides
class StridedBuffer:
    def __init__(self, ptr, shape, strides):
        self.ptr = ptr
        self.shape = shape
        self.strides = strides

    def __getitem__(self, idx):
        offset = sum(i * s for i, s in zip(idx, self.strides))
        return tl.load(self.ptr + offset)

    def __setitem__(self, idx, value):
        offset = sum(i * s for i, s in zip(idx, self.strides))
        tl.store(self.ptr + offset, value)

# Triton kernel for ReLU forward operation on 1D tensors
@triton.jit
def relu_forward_kernel_rank_1(
    input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vec = tl.load(input_ptr + offsets, mask=mask)
    output_vec = tl.where(input_vec > 0, input_vec, 0)
    tl.store(output_ptr + offsets, output_vec, mask=mask)

# Wrapper function for setting up and launching the ReLU forward kernel
def relu_forward_wrapper_rank_1(input_tensor: torch.Tensor, output_tensor: torch.Tensor):
    n_elements = input_tensor.numel()
    tile_size = heuristics_for_tile_size(n_elements)
    num_warps = heuristics_for_num_warps(tile_size)
    grid = (triton.cdiv(n_elements, tile_size),)
    relu_forward_kernel_rank_1[grid](
        input_tensor, output_tensor, n_elements, BLOCK_SIZE=tile_size, num_warps=num_warps
    )

# Example usage
if __name__ == "__main__":
    # Create input and output tensors
    input_tensor = torch.randn(1024, device="cuda")
    output_tensor = torch.empty_like(input_tensor)

    # Run the ReLU forward operation
    relu_forward_wrapper_rank_1(input_tensor, output_tensor)

    # Print the results
    print("Input Tensor:", input_tensor)
    print("Output Tensor:", output_tensor)
