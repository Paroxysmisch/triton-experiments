import torch
import triton
import triton.language as tl

@triton.jit
def fused_layer_norm_relu_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    M, N, K, eps, elementwise_affine,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Pointers to blocks of the input, weight, and output
    m = tl.program_id(0)
    n = tl.program_id(1)

    # Create a block of the input
    input_offset = m * K + tl.arange(0, K)
    weight_offset = n * K + tl.arange(0, K)
    input_block = tl.load(input_ptr + input_offset)
    weight_block = tl.load(weight_ptr + weight_offset)

    # Linear transformation
    result = tl.dot(input_block, weight_block)

    # Add bias if provided
    if bias_ptr is not None:
        result += tl.load(bias_ptr + n)

    # ReLU activation
    result = tl.where(result > 0, result, 0)

    # Layer normalization
    mean = tl.sum(result, axis=0) / N
    var = tl.sum((result - mean) ** 2, axis=0) / N
    result_norm = (result - mean) / tl.sqrt(var + eps)

    if elementwise_affine:
        gamma = tl.load(gamma_ptr + n)
        beta = tl.load(beta_ptr + n)
        result_norm = result_norm * gamma + beta

    # Store the result
    output_offset = m * N + n
    tl.store(output_ptr + output_offset, result_norm)

def fused_layer_norm_relu_linear(input, weight, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True):
    # Check dimensions and allocate output
    M, K = input.shape
    N, _ = weight.shape
    output = torch.empty((M, N), device=input.device, dtype=input.dtype)

    # Allocate mean and variance tensors for layer normalization
    mean = torch.empty((M, N), device=input.device, dtype=input.dtype)
    var = torch.empty((M, N), device=input.device, dtype=input.dtype)

    # Allocate gamma and beta if elementwise affine is enabled
    gamma = torch.ones(N, device=input.device, dtype=input.dtype) if elementwise_affine else None
    beta = torch.zeros(N, device=input.device, dtype=input.dtype) if elementwise_affine else None

    # Launch Triton kernel
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    fused_layer_norm_relu_linear_kernel[grid](
        input, weight, bias, output,
        mean, var, gamma, beta,
        M, N, K, eps, elementwise_affine,
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32
    )

    return output

# Example usage
input = torch.randn(4, 5, device='cuda')
weight = torch.randn(3, 5, device='cuda')
bias = torch.randn(3, device='cuda')
normalized_shape = 3
output = fused_layer_norm_relu_linear(input, weight, bias, normalized_shape)
print(output.shape)  # Expected output shape: (4, 3)
