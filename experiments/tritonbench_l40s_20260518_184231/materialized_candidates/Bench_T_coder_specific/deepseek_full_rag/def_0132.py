import triton
import triton.language as tl
import torch

@triton.jit
def mul_sub_kernel(input, other_mul, other_sub, alpha, out):
    # Triton kernel to perform the multiplication and subtraction
    idx = tl.arange(0, 1)[:, None]
    input_data = tl.load(input + idx)
    other_mul_data = tl.load(other_mul + idx)
    other_sub_data = tl.load(other_sub + idx)
    result = input_data * other_mul_data - alpha * other_sub_data
    tl.store(out + idx, result)

def mul_sub(input, other_mul, other_sub, alpha=1, out=None):
    # Wrapper function to call the Triton kernel
    if out is None:
        out = torch.empty_like(input)
    else:
        out = out.to(input)

    input_flat = input.reshape(-1)
    other_mul_flat = torch.tensor(other_mul, device=input.device).reshape(-1) if not isinstance(other_mul, Number) else other_mul
    other_sub_flat = torch.tensor(other_sub, device=input.device).reshape(-1) if not isinstance(other_sub, Number) else other_sub
    grid = (input_flat.shape[0],)
    mul_sub_kernel[grid](input_flat, other_mul_flat, other_sub_flat, alpha, out=out.reshape(-1))
    return out
