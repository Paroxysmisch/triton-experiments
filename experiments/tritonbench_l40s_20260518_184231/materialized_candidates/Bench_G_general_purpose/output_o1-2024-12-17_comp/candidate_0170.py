import triton
import triton.language as tl
import torch
import math

@triton.jit
def cos_func(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a_value = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_value = tl.cos(a_value)
    tl.store(b_ptr + offsets, b_value, mask=mask)

def cos(a):
    n_elements = a.numel()
    block_size = 1 << int(math.ceil(math.log2(int(n_elements**0.5))))
    grid = ((n_elements + block_size - 1) // block_size,)
    b = torch.empty_like(a)
    cos_func[grid](a, b, n_elements, BLOCK_SIZE=block_size)
    return b
