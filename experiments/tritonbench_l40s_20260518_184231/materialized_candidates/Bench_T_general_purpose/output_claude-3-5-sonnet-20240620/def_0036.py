import triton
import triton.language as tl
import math

@triton.jit
def add_gelu_kernel(
    input_ptr, other_ptr, output_ptr,
    n_elements,
    alpha,
    approximate,
    input_stride,
    other_stride,
    output_stride,
    BLOCK_SIZE: tl.constexpr,
):
    # Global position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input and other
    input_block = tl.load(input_ptr + offsets * input_stride, mask=mask)
    other_block = tl.load(other_ptr + offsets * other_stride, mask=mask)

    # Compute add with alpha
    x = input_block + alpha * other_block

    # GELU computation
    if approximate == 1:  # tanh approximation
        # GELU(x) = 0.5x * (1 + tanh(sqrt(2/π) * (x + 0.044715x³)))
        sqrt_2_pi = 0.7978845608028654  # sqrt(2/pi)
        x3 = x * x * x
        inner = sqrt_2_pi * (x + 0.044715 * x3)
        result = 0.5 * x * (1.0 + tl.tanh(inner))
    else:  # exact computation
        # GELU(x) = x * Φ(x)
        # We approximate the CDF Φ(x) using the error function
        result = 0.5 * x * (1.0 + tl.erf(x / 1.4142135623730951))  # 1/sqrt(2)

    # Store result
    tl.store(output_ptr + offsets * output_stride, result, mask=mask)

def add_gelu(input, other, alpha=1, approximate='none', out=None):
    # Input validation and preparation
    if not input.is_contiguous():
        input = input.contiguous()
    
    # Handle scalar other
    if isinstance(other, (int, float)):
        other = input.new_full(input.size(), other)
    elif not other.is_contiguous():
        other = other.contiguous()
    
    # Prepare output tensor
    if out is None:
        out = input.new_empty(input.size())
    elif not out.is_contiguous():
        out = out.contiguous()
    
    # Validate shapes
    assert input.size() == other.size(), "Input and other must have the same size"
    assert input.size() == out.size(), "Input and output must have the same size"
    
    # Determine grid and block sizes
    BLOCK_SIZE = 1024
    grid = (input.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Convert approximate to integer flag
    approx_flag = 1 if approximate == 'tanh' else 0
    
    # Launch kernel
    add_gelu_kernel[grid](
        input_ptr=input,
        other_ptr=other,
        output_ptr=out,
        n_elements=input.numel(),
        alpha=alpha,
        approximate=approx_flag,
        input_stride=1,
        other_stride=1,
        output_stride=1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out
