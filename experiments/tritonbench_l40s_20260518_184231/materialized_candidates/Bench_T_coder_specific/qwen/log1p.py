import triton
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_X': 256}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_X': 128}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_X': 64}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_X': 32}, num_stages=1, num_warps=4),
    ],
    key=['n_elements']
)
def log1p_forward(x, y, n_elements):
    log1p_kernel[x, y](x, y, n_elements)

def log1p(input, out=None):
    """
    Returns a new tensor with the natural logarithm of (1 + input).

    Args:
        input (Tensor): The input tensor.
        out (Tensor, optional): The output tensor.

    Returns:
        Tensor: A new tensor with the natural logarithm of (1 + input).
    """
    if out is None:
        out = torch.empty_like(input)
    
    grid = lambda meta: (
        triton.cdiv(out.numel(), meta['BLOCK_SIZE_X']),
    )
    
    log1p_forward[grid](input, out, input.numel())
    return out
