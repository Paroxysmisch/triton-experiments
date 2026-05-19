import torch
import triton
import triton.language as tl

# Triton kernel to compute the cosine of each element in the input tensor
@triton.jit
def cos_func(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.cos(x)
    tl.store(y_ptr + offsets, y, mask=mask)

# Wrapper function to compute the cosine of each element in the input tensor
def cos(x):
    y = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    cos_func[grid](x, y, n_elements, BLOCK_SIZE=1024)
    return y

# Triton kernel to compute the sign bit of each cosine result
@triton.jit
def signbit_func(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.signbit(x)
    tl.store(y_ptr + offsets, y, mask=mask)

# Wrapper function to compute the sign bit of each cosine result
def signbit(x):
    y = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    signbit_func[grid](x, y, n_elements, BLOCK_SIZE=1024)
    return y

# Main function to compute the cosine and sign bit of each element in the input tensor
def cos_signbit(input):
    cos_result = cos(input)
    sign_bit = signbit(cos_result)
    return cos_result, sign_bit
