import triton
import triton.language as tl
import torch

# Constants for SELU
ALPHA = 1.6732632423543772848170429916717
SCALE = 1.0507009873554804934193349852946

# Triton kernel for SELU
@triton.jit
def _selu_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    
    # Apply SELU formula
    pos = tl.maximum(x, 0)
    neg = tl.minimum(x, 0)
    exp_neg = tl.exp(neg / ALPHA)
    y = SCALE * (pos + neg * exp_neg)
    
    tl.store(Y + offsets, y, mask=mask)

# Wrapper function for SELU
def selu(input, inplace=False):
    if inplace:
        output = input
    else:
        output = input.clone()
    
    N = input.numel()
    grid_size = (N + 255) // 256
    
    _selu_kernel[(grid_size,)](input, output, N, BLOCK_SIZE=256)
    
    return output

# Example usage
if __name__ == "__main__":
    x = torch.randn(1024, device='cuda')
    y = selu(x)
    print(y)
