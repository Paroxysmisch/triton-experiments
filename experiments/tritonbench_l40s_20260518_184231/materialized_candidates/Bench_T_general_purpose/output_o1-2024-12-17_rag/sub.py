import triton
import triton.language as tl
import math
import torch

@triton.jit
def _sub_kernel(
    a_ptr, 
    b_ptr, 
    c_ptr, 
    alpha, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr, 
    has_scalar_b: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a_val = tl.load(a_ptr + offsets, mask=mask)
    if has_scalar_b:
        c_val = a_val - alpha * b_ptr  # b_ptr holds scalar value in this mode
    else:
        b_val = tl.load(b_ptr + offsets, mask=mask)
        c_val = a_val - alpha * b_val
    tl.store(c_ptr + offsets, c_val, mask=mask)

def sub(input, other, *, alpha=1, out=None):
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a Tensor.")
    if not torch.is_tensor(other) and not isinstance(other, (int, float, complex)):
        raise TypeError("other must be a Tensor or a Number.")
    # Broadcast to common shape
    if torch.is_tensor(other):
        input_b, other_b = torch.broadcast_tensors(input, other)
    else:
        input_b = input
        other_b = other  # scalar path
    # Prepare output
    if out is None:
        if torch.is_tensor(other_b):
            out = torch.empty_like(input_b)
        else:
            out = torch.empty_like(input_b, dtype=input_b.dtype)
    # Allocate contiguous copies for kernel usage
    a_contig = input_b.contiguous()
    if torch.is_tensor(other_b):
        b_contig = other_b.contiguous()
    # Determine kernel grid
    n_elements = a_contig.numel()
    block_size = triton.next_power_of_2(int(math.sqrt(n_elements)))
    grid = ( (n_elements + block_size - 1) // block_size, )
    # Launch kernel
    if torch.is_tensor(other_b):
        _sub_kernel[grid](
            a_contig, 
            b_contig, 
            out, 
            alpha, 
            n_elements, 
            block_size, 
            has_scalar_b=False
        )
    else:
        # Store the scalar in b_ptr argument directly
        _sub_kernel[grid](
            a_contig, 
            other_b, 
            out, 
            alpha, 
            n_elements, 
            block_size, 
            has_scalar_b=True
        )
    return out
