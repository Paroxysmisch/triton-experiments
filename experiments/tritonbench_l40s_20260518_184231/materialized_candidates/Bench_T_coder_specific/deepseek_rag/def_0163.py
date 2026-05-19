import torch
import triton
import triton.language as tl
import math

@triton.jit
def cos_func(a, b, n_elements, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    a_value = tl.load(a + offset, mask=mask)
    b_value = tl.cos(a_value.to(tl.float32))
    tl.store(b + offset, b_value, mask=mask)

def cos(A):
    B = torch.empty_like(A)
    n_elements = A.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)
    cos_func[(grid_size, 1, 1)](A, B, n_elements, block_size)
    return B

def cos_signbit(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cos_result = cos(input)
    sign_bit = torch.signbit(cos_result)
    return cos_result, sign_bit
