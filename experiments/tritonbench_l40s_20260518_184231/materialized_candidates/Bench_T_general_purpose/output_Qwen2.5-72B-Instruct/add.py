import triton
import triton.language as tl

@triton.jit
def add_kernel(
    input_ptr,  # Pointer to the input tensor
    other_ptr,  # Pointer to the other tensor or number
    out_ptr,    # Pointer to the output tensor
    input_n,    # Size of the input tensor
    other_n,    # Size of the other tensor
    alpha,      # Scalar multiplier for other
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_n

    input_vec = tl.load(input_ptr + offsets, mask=mask)
    other_vec = tl.load(other_ptr + (offsets % other_n), mask=mask)  # Broadcasting

    out_vec = input_vec + alpha * other_vec
    tl.store(out_ptr + offsets, out_vec, mask=mask)

import torch
import triton
import triton.language as tl

def add(input, other, *, alpha=1, out=None):
    # Ensure input and other are tensors
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input, device='cuda')
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, device='cuda')

    # Ensure input and other have the same device
    assert input.device == other.device, "input and other must be on the same device"

    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input, device=input.device)
    else:
        assert out.device == input.device, "out must be on the same device as input"

    # Determine the sizes
    input_n = input.numel()
    other_n = other.numel()

    # Determine the grid and block sizes
    BLOCK_SIZE = 1024
    grid_size = (input_n + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    add_kernel[grid_size, BLOCK_SIZE](
        input, other, out, input_n, other_n, alpha
    )

    return out
