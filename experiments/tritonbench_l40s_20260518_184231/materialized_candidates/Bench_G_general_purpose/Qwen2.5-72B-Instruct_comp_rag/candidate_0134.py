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

# Wrapper function to set up and launch the kernel
def add_wrapper(x, y):
    out = torch.zeros_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    add_kernel[(num_blocks,)](x, y, out, n_elements, BLOCK_SIZE)
    return out

# Example usage
if __name__ == "__main__":
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
    y = torch.tensor([5.0, 6.0, 7.0, 8.0], device='cuda')
    result = add_wrapper(x, y)
    print(result)  # Output: tensor([6., 8., 10., 12.], device='cuda:0')
