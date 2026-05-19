import triton
import triton.language as tl
import torch
from typing import Optional

class StridedBuffer:
    """Handles tensors with custom strides, including negative strides."""
    def __init__(self, tensor: torch.Tensor):
        assert tensor.is_cuda, "Tensor must be on CUDA device"
        self.device = tensor.device
        self.dtype = tensor.dtype
        self.element_size = tensor.element_size()
        self.stride = tensor.stride(0)
        self.ptr = tensor.data_ptr()

        # Adjust for negative strides
        if self.stride < 0:
            self.ptr += self.stride * (tensor.size(0) - 1) * self.element_size
            self.stride = abs(self.stride)

def heuristics_for_tile_size(n_elements: int, max_tile: int = 1024) -> int:
    """Determine optimal tile size considering input dimensions and max allowed."""
    return min(max_tile, triton.next_power_of_2(n_elements)) if n_elements > 0 else 128

def heuristics_for_num_warps(tile_size: int) -> int:
    """Calculate number of warps based on tile size."""
    return min(max(tile_size // 256, 1), 8)

@triton.jit
def relu_forward_kernel_rank_1(
    input_ptr,
    output_ptr,
    n_elements,
    input_stride,
    output_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for ReLU forward pass on 1D tensors."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Calculate memory offsets using strides
    input_offsets = offsets * input_stride
    output_offsets = offsets * output_stride

    # Load data, apply ReLU, and store result
    x = tl.load(input_ptr + input_offsets, mask=mask)
    zero = tl.zeros_like(x)
    result = tl.where(x >= 0, x, zero)
    tl.store(output_ptr + output_offsets, result, mask=mask)

def relu_forward_wrapper_rank_1(input: torch.Tensor, output: torch.Tensor):
    """Launch the ReLU kernel with optimal configuration."""
    n_elements = input.numel()
    if n_elements == 0:
        return  # Handle empty tensor

    # Configure kernel launch parameters
    tile_size = heuristics_for_tile_size(n_elements)
    num_warps = heuristics_for_num_warps(tile_size)
    grid = (triton.cdiv(n_elements, tile_size),)

    # Create strided buffers for memory access
    input_buf = StridedBuffer(input)
    output_buf = StridedBuffer(output)

    # Launch kernel
    relu_forward_kernel_rank_1[grid](
        input_buf.ptr,
        output_buf.ptr,
        n_elements,
        input_buf.stride,
        output_buf.stride,
        BLOCK_SIZE=tile_size,
        num_warps=num_warps,
    )

# Example usage
def relu(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """User-facing ReLU function with similar interface to torch.relu."""
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)
    
    relu_forward_wrapper_rank_1(input, output)
    return output
