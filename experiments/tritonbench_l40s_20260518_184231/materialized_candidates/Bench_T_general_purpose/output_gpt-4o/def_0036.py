import triton
import triton.language as tl
import torch

@triton.jit
def add_gelu_kernel(
    input_ptr, other_ptr, output_ptr, alpha, N, approximate, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    other_vals = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    
    # Compute the addition and scaling
    result = input_vals + alpha * other_vals
    
    # Apply GELU activation
    if approximate == 0:  # 'none'
        # Exact GELU using CDF of Gaussian
        sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
        cdf = 0.5 * (1.0 + tl.erf(result / tl.sqrt(2.0)))
        gelu_result = result * cdf
    else:  # 'tanh'
        # Approximate GELU using tanh
        sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
        tanh_arg = sqrt_2_over_pi * (result + 0.044715 * result * result * result)
        gelu_result = 0.5 * result * (1.0 + tl.tanh(tanh_arg))
    
    tl.store(output_ptr + offsets, gelu_result, mask=mask)

def add_gelu(input, other, alpha=1, approximate='none', out=None):
    assert approximate in ['none', 'tanh'], "approximate must be 'none' or 'tanh'"
    
    # Determine the size of the input
    N = input.numel()
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(input)
    
    # Ensure other is a tensor
    if isinstance(other, (int, float)):
        other = torch.full_like(input, other)
    else:
        assert other.shape == input.shape, "other must be the same shape as input or a scalar"
    
    # Set approximate mode
    approximate_mode = 0 if approximate == 'none' else 1
    
    # Launch the Triton kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    add_gelu_kernel[grid](
        input_ptr=input.data_ptr(),
        other_ptr=other.data_ptr(),
        output_ptr=out.data_ptr(),
        alpha=alpha,
        N=N,
        approximate=approximate_mode,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
