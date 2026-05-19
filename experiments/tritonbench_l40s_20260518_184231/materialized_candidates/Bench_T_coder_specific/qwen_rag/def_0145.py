import triton
import triton.language as tl
import torch

# Define the constants for the derivatives of the digamma function
DIGAMMA_DERIVATIVES = [
    lambda x: -torch.polygamma(0, x),  # First derivative of digamma (digamma itself)
    lambda x: torch.polygamma(1, x),  # Second derivative of digamma
    lambda x: torch.polygamma(2, x),  # Third derivative of digamma
    lambda x: torch.polygamma(3, x),  # Fourth derivative of digamma
    lambda x: torch.polygamma(4, x),  # Fifth derivative of digamma
    lambda x: torch.polygamma(5, x),  # Sixth derivative of digamma
    lambda x: torch.polygamma(6, x),  # Seventh derivative of digamma
    lambda x: torch.polygamma(7, x),  # Eighth derivative of digamma
    lambda x: torch.polygamma(8, x),  # Ninth derivative of digamma
    lambda x: torch.polygamma(9, x),  # Tenth derivative of digamma
]

# Triton kernel to compute the n-th derivative of the digamma function
@triton.jit
def _polygamma_kernel(X, OUT, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    grid_size = tl.cdiv(N, BLOCK_SIZE)
    
    for i in range(grid_size):
        idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = X[idx]
        n = tl.load(idx, mask=idx < N, other=0)
        d = DIGAMMA_DERIVATIVES[n](x)
        tl.store(OUT + pid * BLOCK_SIZE, d, mask=idx < N)

# Wrapper function to call the Triton kernel
def polygamma(n, input, *, out=None) -> torch.Tensor:
    assert isinstance(n, int) and n >= 0, "n must be a nonnegative integer"
    assert input.is_cuda, "Input tensor must be on CUDA"
    
    if out is None:
        out = input.new_empty_like(input)
    
    N = input.numel()
    BLOCK_SIZE = 256
    
    _polygamma_kernel[(N // BLOCK_SIZE,)](input, out, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

# Example usage
if __name__ == "__main__":
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device='cuda')
    n = 1
    result = polygamma(n, x)
    print(result)
