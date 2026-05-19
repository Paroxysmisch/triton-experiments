import triton
import triton.language as tl
import torch
import math

@triton.jit
def exp_func(a, b, n_elements, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    a_value = tl.load(a + offset, mask=mask)
    b_value = tl.exp(a_value.to(tl.float32))
    tl.store(b + offset, b_value, mask=mask)

def exp(A):
    B = torch.empty_like(A)
    n_elements = A.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    exp_func[(grid_size, 1, 1)](A, B, n_elements, block_size)
    return B

def test_exp():
    torch.manual_seed(0)
    a = torch.rand(128, 128)
    b = exp(a)
    c = torch.exp(a)
    assert torch.allclose(b, c)
