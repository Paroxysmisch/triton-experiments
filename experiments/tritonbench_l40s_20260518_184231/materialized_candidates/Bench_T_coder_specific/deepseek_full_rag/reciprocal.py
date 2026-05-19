import triton
import triton.language as tl
import torch
import math

@triton.jit
def reciprocal_func(a, b, n_elements, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    a_value = tl.load(a + offset, mask=mask)
    b_value = 1 / a_value.to(tl.float32)
    tl.store(b + offset, b_value, mask=mask)  

def reciprocal(A):
    B = torch.empty_like(A)
    n_elements = A.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    reciprocal_func[(grid_size, 1, 1)](A, B, n_elements, block_size)
    return B

if __name__ == "__main__":
    a = torch.randn([4, 4], device="cuda")
    print(torch.reciprocal(a))
    print(reciprocal(a))
