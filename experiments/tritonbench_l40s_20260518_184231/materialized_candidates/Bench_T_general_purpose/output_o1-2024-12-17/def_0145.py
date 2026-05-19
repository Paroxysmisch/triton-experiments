import triton
import triton.language as tl
import math
import torch

@triton.jit
def _polygamma_kernel(
    x_ptr, 
    out_ptr,
    n,
    factorial_n, 
    numel,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    base_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = base_idx < numel
    
    x_vals = tl.load(x_ptr + base_idx, mask=mask)
    # Initialize result
    result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Partial sum for polygamma(n,x)
    # polygamma(n, x) = sum_{k=0}^∞ [((-1)^n * n!) / (x + k)^(n+1)]
    # Here we truncate for demonstration
    MAX_ITER = 50
    sign = tl.where((n % 2) == 0, 1.0, -1.0)
    for k in range(MAX_ITER):
        denom = (x_vals + k)
        term = sign * factorial_n / (denom ** (n + 1))
        result += term
    
    tl.store(out_ptr + base_idx, result, mask=mask)

def polygamma(n, input, *, out=None) -> torch.Tensor:
    """
    n (int): the order of the polygamma function.
    input (Tensor): the input tensor.
    out (Tensor, optional): the output tensor.
    """
    if n < 0 or not isinstance(n, int):
        raise ValueError("n must be a nonnegative integer.")
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor.")
    
    # Prepare output
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32)
    else:
        if out.shape != input.shape:
            raise ValueError("out must have the same shape as input.")
    
    # Compute factorial(n)
    factorial_n = float(math.factorial(n))
    
    # Launch Triton kernel
    numel = input.numel()
    BLOCK_SIZE = 1024
    grid = ( (numel + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    _polygamma_kernel[grid](
        input.data_ptr(),
        out.data_ptr(),
        n,
        factorial_n,
        numel,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out
