import triton
import triton.language as tl
import torch

@triton.jit
def mul_sub_kernel(input_ptr, other_mul_ptr, other_sub_ptr, alpha, output_ptr, N):
    pid = tl.program_id(0)
    offset = pid * tl.constexpr(1)
    mask = offset < N
    input_val = tl.load(input_ptr + offset, mask=mask)
    other_mul_val = tl.load(other_mul_ptr + offset, mask=mask)
    other_sub_val = tl.load(other_sub_ptr + offset, mask=mask)
    result = (input_val * other_mul_val) - (alpha * other_sub_val)
    tl.store(output_ptr + offset, result, mask=mask)

def mul_sub(input, other_mul, other_sub, alpha=1, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"

    input_ptr = triton.pointers.allocate_shared(input)
    other_mul_ptr = triton.pointers.allocate_shared(other_mul)
    other_sub_ptr = triton.pointers.allocate_shared(other_sub)
    output_ptr = triton.pointers.allocate_shared(out)

    N = input.numel()
    grid = lambda meta: (triton.cdiv(N, meta['n_elements']),)
    mul_sub_kernel[grid](input_ptr, other_mul_ptr, other_sub_ptr, alpha, output_ptr, N)

    return out
