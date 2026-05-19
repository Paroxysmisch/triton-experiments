import triton
import triton.language as tl

@triton.jit
def add_gelu_kernel(
    input_ptr, other_ptr, out_ptr, alpha, approximate, size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    input_values = tl.load(input_ptr + offsets, mask=mask)
    other_values = tl.load(other_ptr + offsets, mask=mask)

    # Scale other by alpha
    other_values *= alpha

    # Add input and scaled other
    result = input_values + other_values

    # Apply GELU
    if approximate == 0:  # 'none'
        result = result * tl.erf(result / tl.sqrt(tl.float32(2.0))) * tl.float32(0.5) + result * tl.float32(0.5)
    elif approximate == 1:  # 'tanh'
        cdf = tl.tanh(tl.sqrt(tl.float32(2.0) / tl.float32(3.141592653589793)) * (result + tl.float32(0.044715) * result * result * result))
        result = result * (tl.float32(0.5) * (tl.float32(1.0) + cdf))

    tl.store(out_ptr + offsets, result, mask=mask)

import torch
import triton
import triton.language as tl

def add_gelu(input, other, alpha=1, approximate='none', out=None):
    if out is None:
        out = torch.empty_like(input)

    # Determine the size of the input tensor
    size = input.numel()

    # Convert approximate string to integer for kernel
    approximate_map = {'none': 0, 'tanh': 1}
    approximate_int = approximate_map[approximate]

    # Ensure other is a tensor
    if not isinstance(other, torch.Tensor):
        other = torch.full_like(input, other)

    # Launch the Triton kernel
    grid = (triton.cdiv(size, 1024),)
    add_gelu_kernel[grid](
        input, other, out, alpha, approximate_int, size, BLOCK_SIZE=1024
    )

    return out
