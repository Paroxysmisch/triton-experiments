import triton
import triton.language as tl
import torch

# Constants for SELU
ALPHA = 1.6732632423543772848170429916717
SCALE = 1.0507009873554804934193349852946

# Triton kernel for SELU
@triton.jit
def _selu(input, output, stride, N):
    idx = tl.program_id(0) * tl.block_dim() + tl.arange(0, tl.block_dim())
    mask = idx < N
    x = tl.load(input + idx, mask=mask)
    
    # SELU computation
    selu_result = SCALE * (tl.maximum(x, 0) + tl.minimum(ALPHA * (tl.exp(x) - 1), 0))
    
    # Store the result
    tl.store(output + idx, selu_result, mask=mask)

# Wrapper function for SELU
def selu(input, inplace=False):
    assert input.is_cuda
    N = input.numel()
    output = input if inplace else input.new_empty(input.shape, dtype=input.dtype)
    
    # Launch the Triton kernel
    _selu[(N,)](input, output, input.stride(0), N)
    
    return output
