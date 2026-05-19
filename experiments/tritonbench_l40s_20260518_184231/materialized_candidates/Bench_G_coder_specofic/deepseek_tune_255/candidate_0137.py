import torch
import triton
import triton.language as tl

# Triton kernel for element-wise addition
@triton.jit
def add_kernel(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)

# Wrapper function for calling the Triton kernel
def add_wrapper(x, y):
    out = torch.zeros_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    add_kernel[(num_blocks,)](x, y, out, n_elements, BLOCK_SIZE)
    return out

# Example usage
x = torch.randn(10000, device='cuda')
y = torch.randn(10000, device='cuda')
out_triton = add_wrapper(x, y)
out_torch = x + y
print(out_torch)
print(out_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(out_torch - out_triton))}')
