import triton
import triton.language as tl
import torch
import math

# Kernel for bitwise AND operation on two tensors
@triton.jit
def bitwise_and_func_tensor(A_ptr, B_ptr, C_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    A = tl.load(A_ptr + offsets, mask=mask)
    B = tl.load(B_ptr + offsets, mask=mask)
    C = A & B
    tl.store(C_ptr + offsets, C, mask=mask)

# Kernel for bitwise AND operation on a tensor A and a constant scalar B
@triton.jit
def bitwise_and_func_scalar(A_ptr, B: tl.constexpr, C_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    A = tl.load(A_ptr + offsets, mask=mask)
    C = A & B
    tl.store(C_ptr + offsets, C, mask=mask)

# Wrapper function matching the provided signature
def bitwise_and(input, other, *, out=None):
    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input)
    n_elements = out.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)

    if isinstance(other, torch.Tensor):
        # Ensure input and other have the same shape
        assert input.shape == other.shape, "Input tensors must have the same shape."
        # Call kernel for tensor-tensor bitwise AND
        bitwise_and_func_tensor[(grid_size,)](input, other, out, n_elements, block_size)
    else:
        # Call kernel for tensor-scalar bitwise AND
        bitwise_and_func_scalar[(grid_size,)](input, other, out, n_elements, block_size)
    
    return out
