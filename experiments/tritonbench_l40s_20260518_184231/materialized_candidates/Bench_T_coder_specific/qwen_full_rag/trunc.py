import torch
import triton
import triton.language as tl

@triton.jit
def trunc(input, *, out=None):
    casted = input.to(tl.int64)
    return casted.to(input.dtype)

# Test the function
input_tensor = torch.tensor([1.5, -2.3, 3.7], device='cuda')
output_tensor = trunc(input_tensor)
print(output_tensor)  # Expected: tensor([1, -2, 3], device='cuda')

input_tensor_int = torch.tensor([1, -2, 3], device='cuda', dtype=torch.int64)
output_tensor_int = trunc(input_tensor_int)
print(output_tensor_int)  # Expected: tensor([1, -2, 3], device='cuda', dtype=torch.int64)
