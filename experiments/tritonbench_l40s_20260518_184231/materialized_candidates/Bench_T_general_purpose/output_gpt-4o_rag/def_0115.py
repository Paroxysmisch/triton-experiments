import torch
import triton
import triton.language as tl

@triton.jit
def kernel_tanh_linear(
    output_ptr,  # Pointer to the output tensor
    input_ptr,   # Pointer to the input tensor
    weight_ptr,  # Pointer to the weight matrix
    bias_ptr,    # Pointer to the bias vector
    M, N, K,     # Dimensions: M = batch size, N = out_features, K = in_features
    stride_im, stride_in,  # Strides for input
    stride_wn, stride_wk,  # Strides for weight
    stride_om, stride_on,  # Strides for output
    has_bias: tl.constexpr,  # Whether bias is provided
):
    # Compute program ID and ranges
    pid = tl.program_id(axis=0)
    block_m = tl.arange(0, M)
    block_n = tl.arange(0, N)
    
    # Compute input and weight offsets
    input_offset = block_m[:, None] * stride_im + tl.arange(0, K)[None, :] * stride_in
    weight_offset = tl.arange(0, K)[:, None] * stride_wk + block_n[None, :] * stride_wn
    
    # Load input and weight data
    input_matrix = tl.load(input_ptr + input_offset)
    weight_matrix = tl.load(weight_ptr + weight_offset)
    
    # Perform matrix multiplication
    output_matrix = tl.dot(input_matrix, weight_matrix)
    
    # Add bias if provided
    if has_bias:
        bias = tl.load(bias_ptr + block_n)
        output_matrix += bias[None, :]
    
    # Apply Tanh activation function
    output_matrix = tl.tanh(output_matrix)
    
    # Compute output offset and store the result
    output_offset = block_m[:, None] * stride_om + block_n[None, :] * stride_on
    tl.store(output_ptr + output_offset, output_matrix)

def tanh_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    # Reshape input tensor
    batch_shape, in_features = input.shape[:-1], input.shape[-1]
    out_features = weight.shape[0]
    input_reshaped = input.reshape(-1, in_features)
    
    # Prepare output tensor
    output = torch.empty((input_reshaped.shape[0], out_features), device=input.device, dtype=input.dtype)
    
    # Determine if bias is provided
    has_bias = bias is not None
    
    # Launch Triton kernel
    grid = lambda META: (triton.cdiv(input_reshaped.shape[0], META['BLOCK_M']),)
    kernel_tanh_linear[grid](
        output, input_reshaped, weight, bias if has_bias else input_reshaped,  # Pointers
        input_reshaped.shape[0], out_features, in_features,  # Dimensions
        input_reshaped.stride(0), input_reshaped.stride(1),  # Strides for input
        weight.stride(0), weight.stride(1),  # Strides for weight
        output.stride(0), output.stride(1),  # Strides for output
        has_bias=has_bias  # Meta-parameter for bias
    )
    
    # Reshape output to match input batch shape
    return output.reshape(*batch_shape, out_features)
