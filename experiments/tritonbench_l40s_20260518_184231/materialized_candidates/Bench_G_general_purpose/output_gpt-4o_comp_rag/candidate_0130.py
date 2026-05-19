import triton
import triton.language as tl
import torch

# Triton kernel for element-wise addition
@triton.jit
def add_kernel(in_ptr0, in_ptr1, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    result = x + y
    tl.store(out_ptr + offsets, result, mask=mask)

# Wrapper function to set up and launch the kernel
def add_wrapper(x, y, BLOCK_SIZE=1024):
    assert x.shape == y.shape, "Input tensors must have the same shape"
    n_elements = x.numel()
    out = torch.zeros_like(x)
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    add_kernel[(num_blocks,)](x, y, out, n_elements, BLOCK_SIZE)
    return out

# Example usage
if __name__ == "__main__":
    # Create input tensors
    x = torch.randn(1024, device='cuda', dtype=torch.float32)
    y = torch.randn(1024, device='cuda', dtype=torch.float32)

    # Perform element-wise addition
    result = add_wrapper(x, y)

    # Print result
    print(result)
