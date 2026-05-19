import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(input_ptr, other_ptr, output_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    other_vals = tl.load(other_ptr + offsets, mask=mask)
    output_vals = input_vals + alpha * other_vals
    tl.store(output_ptr + offsets, output_vals, mask=mask)

def add(input, other, alpha=1, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape

    n_elements = input.numel()
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)

    if isinstance(other, torch.Tensor):
        other_ptr = other.data_ptr()
    else:
        other_ptr = other

    input_ptr = input.data_ptr()
    output_ptr = out.data_ptr()

    add_kernel[(grid_size, BLOCK_SIZE)](input_ptr, other_ptr, output_ptr, alpha, n_elements, BLOCK_SIZE)

    return out
