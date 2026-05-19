import torch
import triton
import triton.language as tl

# Triton kernel for GELU using the exact formula (erf approximation)
@triton.jit
def gelu_none_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)
    # Compute the GELU function using the error function approximation
    x_gelu = 0.5 * x_fp32 * (1 + tl.math.erf(x_fp32 * 0.7071067811))
    tl.store(Y + offsets, x_gelu.to(x.dtype), mask=mask)

# Triton kernel for GELU using the tanh approximation
@triton.jit
def gelu_tanh_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    x_fp32 = x.to(tl.float32)
    # Compute the GELU function using the tanh approximation
    x_gelu = 0.5 * x_fp32 * (1 + tl.math.tanh(0.79788456 * x_fp32 * (1 + 0.044715 * x_fp32 * x_fp32)))
    tl.store(Y + offsets, x_gelu.to(x.dtype), mask=mask)

# Wrapper function for the GELU activation
def gelu(input, approximate='none'):
    # Ensure the input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")
    
    # Ensure the input is on the same device
    device = input.device
    dtype = input.dtype
    N = input.numel()
    
    # Allocate output tensor
    output = torch.empty_like(input, device=device, dtype=dtype)
    
    # Determine the block size
    BLOCK_SIZE = 1024
    
    # Choose the appropriate kernel based on the 'approximate' argument
    if approximate == 'none':
        gelu_kernel = gelu_none_kernel
    elif approximate == 'tanh':
        gelu_kernel = gelu_tanh_kernel
    else:
        raise ValueError(f"Invalid approximate value: {approximate}")
    
    # Launch the kernel
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    gelu_kernel[grid](input, output, N, BLOCK_SIZE)
    
    return output

# Example usage
if __name__ == "__main__":
    # Create a random input tensor
    input_tensor = torch.randn(1024, device='cuda')
    
    # Apply GELU with exact formula
    output_exact = gelu(input_tensor, approximate='none')
    
    # Apply GELU with tanh approximation
    output_approx = gelu(input_tensor, approximate='tanh')
    
    print("Exact GELU output:", output_exact)
    print("Approximate GELU output:", output_approx)
