import triton
import triton.language as tl
import torch

@triton.jit
def chebyshev_kernel(
    input_ptr,
    output_ptr,
    n_value,
    input_numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    for i in range(0, BLOCK_SIZE):
        idx = pid * BLOCK_SIZE + i
        if idx >= input_numel:
            return
        x = tl.load(input_ptr + idx)
        if n_value == 0:
            res = 1.0
        elif n_value == 1:
            res = x
        else:
            abs_x = tl.abs(x)
            if n_value < 6 or abs_x > 1.0:
                t_prev_prev = 1.0
                t_prev = x
                for k in range(2, n_value + 1):
                    t_current = 2.0 * x * t_prev - t_prev_prev
                    t_prev_prev, t_prev = t_prev, t_current
                res = t_prev
            else:
                theta = tl.math.acos(x)
                res = tl.math.cos(n_value * theta)
        tl.store(output_ptr + idx, res)

def chebyshev_polynomial_t(input: torch.Tensor, n: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    if n.numel() != 1:
        raise ValueError("n must be a scalar tensor")
    n_value = n.item()
    if not isinstance(n_value, int):
        raise ValueError("n must be an integer tensor")
    if n_value < 0:
        raise ValueError("n must be a non-negative integer")
    
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("out tensor must have the same shape as input")
    
    input_flat = input.view(-1)
    out_flat = out.view(-1)
    numel = input_flat.numel()
    
    if numel == 0:
        return out
    
    BLOCK_SIZE = 128
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    chebyshev_kernel[grid](input_flat, out_flat, n_value, numel, BLOCK_SIZE=BLOCK_SIZE)
    
    return out
