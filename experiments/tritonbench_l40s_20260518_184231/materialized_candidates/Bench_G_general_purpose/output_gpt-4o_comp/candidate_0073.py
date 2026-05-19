import triton
import triton.language as tl

# Define a utility function to calculate mean and variance
@triton.jit
def compute_mean_var(x, N):
    mean = tl.sum(x, axis=0) / N
    var = tl.sum((x - mean) ** 2, axis=0) / N
    return mean, var

# Forward kernel for layer normalization
@triton.autotune(configs=[
    triton.Config({'BLOCK_N': 128}, num_warps=4),
    triton.Config({'BLOCK_N': 256}, num_warps=8),
])
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    data_ptr, weight_ptr, bias_ptr, residual_ptr, output_ptr,
    M, N, use_bias, use_residual,
    BLOCK_N: tl.constexpr
):
    # Compute the offsets
    row_idx = tl.program_id(0)
    data_offset = row_idx * N
    weight_offset = 0
    bias_offset = 0

    # Load data
    data = tl.load(data_ptr + data_offset + tl.arange(0, BLOCK_N))
    weight = tl.load(weight_ptr + weight_offset + tl.arange(0, BLOCK_N))
    if use_bias:
        bias = tl.load(bias_ptr + bias_offset + tl.arange(0, BLOCK_N))
    if use_residual:
        residual = tl.load(residual_ptr + data_offset + tl.arange(0, BLOCK_N))

    # Compute mean and variance
    mean, var = compute_mean_var(data, N)

    # Normalize
    normalized = (data - mean) / tl.sqrt(var + 1e-5)

    # Apply weights and bias
    output = normalized * weight
    if use_bias:
        output += bias
    if use_residual:
        output += residual

    # Store result
    tl.store(output_ptr + data_offset + tl.arange(0, BLOCK_N), output)

# Backward kernel for layer normalization
@triton.autotune(configs=[
    triton.Config({'BLOCK_N': 128}, num_warps=4),
    triton.Config({'BLOCK_N': 256}, num_warps=8),
])
@triton.jit
def _layer_norm_bwd_kernel(
    grad_output_ptr, data_ptr, weight_ptr, bias_ptr, grad_data_ptr, grad_weight_ptr, grad_bias_ptr,
    M, N, use_bias, use_residual,
    BLOCK_N: tl.constexpr
):
    # Compute the offsets
    row_idx = tl.program_id(0)
    data_offset = row_idx * N
    weight_offset = 0
    bias_offset = 0

    # Load data
    grad_output = tl.load(grad_output_ptr + data_offset + tl.arange(0, BLOCK_N))
    data = tl.load(data_ptr + data_offset + tl.arange(0, BLOCK_N))
    weight = tl.load(weight_ptr + weight_offset + tl.arange(0, BLOCK_N))
    if use_bias:
        bias = tl.load(bias_ptr + bias_offset + tl.arange(0, BLOCK_N))

    # Compute mean and variance
    mean, var = compute_mean_var(data, N)

    # Normalize
    normalized = (data - mean) / tl.sqrt(var + 1e-5)

    # Compute gradients
    grad_normalized = grad_output * weight
    grad_data = grad_normalized / tl.sqrt(var + 1e-5)
    grad_weight = tl.sum(grad_output * normalized, axis=0)
    if use_bias:
        grad_bias = tl.sum(grad_output, axis=0)

    # Store gradients
    tl.store(grad_data_ptr + data_offset + tl.arange(0, BLOCK_N), grad_data)
    tl.store(grad_weight_ptr + weight_offset + tl.arange(0, BLOCK_N), grad_weight)
    if use_bias:
        tl.store(grad_bias_ptr + bias_offset + tl.arange(0, BLOCK_N), grad_bias)

# Example wrapper function
def layer_norm_fwd(data, weight, bias=None, residual=None, use_bias=True, use_residual=False):
    M, N = data.shape
    output = torch.empty_like(data)
    grid = (M,)
    _layer_norm_fwd_1pass_kernel[grid](
        data, weight, bias, residual, output, M, N, use_bias, use_residual
    )
    return output

def layer_norm_bwd(grad_output, data, weight, bias=None, use_bias=True, use_residual=False):
    M, N = data.shape
    grad_data = torch.empty_like(data)
    grad_weight = torch.empty_like(weight)
    grad_bias = torch.empty_like(bias) if use_bias else None
    grid = (M,)
    _layer_norm_bwd_kernel[grid](
        grad_output, data, weight, bias, grad_data, grad_weight, grad_bias, M, N, use_bias, use_residual
    )
    return grad_data, grad_weight, grad_bias
