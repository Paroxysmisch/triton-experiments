import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    in_ptr0, in_ptr1, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)

def add_wrapper(x: torch.Tensor, y: torch.Tensor, BLOCK_SIZE=1024):
    out = torch.zeros_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return out  # Handle empty tensor case
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

# Example usage
if __name__ == "__main__":
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    y = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    result = add_wrapper(x, y)
    print(result)  # Output: tensor([5., 7., 9.], device='cuda:0')
