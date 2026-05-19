import triton
import torch

@triton.jit
def scaled_add_dot_kernel(y_ptr, x_ptr, alpha_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offset)
    alpha = tl.load(alpha_ptr)
    y = tl.load(y_ptr + offset)
    tl.store(y_ptr + offset, y + alpha * x)
    return tl.dot(y_ptr + offset, y_ptr + offset)

def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float):
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(y.numel(), meta['BLOCK_SIZE']),)
    alpha_tensor = torch.tensor([alpha]).to(y.device)
    result = scaled_add_dot_kernel[grid](y, x, alpha_tensor, BLOCK_SIZE)
    return result
