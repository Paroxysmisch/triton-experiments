import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=1, num_warps=4),
    ],
    key=['n_elements']
)
def _floor_forward(input, out, BLOCK_SIZE):
    n_elements = input.numel()
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    floor_kernel[grid_size, BLOCK_SIZE](input.data_ptr(), out.data_ptr(), n_elements, BLOCK_SIZE)

def floor(input, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert input.shape == out.shape, "Input and output tensors must have the same shape."
    
    # Call the Triton kernel
    _floor_forward(input, out)
    
    return out
