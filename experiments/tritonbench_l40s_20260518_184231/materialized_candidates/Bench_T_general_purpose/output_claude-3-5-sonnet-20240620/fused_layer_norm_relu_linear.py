import triton
import triton.language as tl

@triton.jit
def fused_layer_norm_relu_linear_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr, 
                                        normalized_shape, eps, elementwise_affine, 
                                        n_elements: tl.constexpr):
    # Load input tensor
    input_tensor = tl.load(input_ptr + tl.arange(0, n_elements) * input_tensor.stride(0))
    
    # Linear transformation
    weight = tl.load(weight_ptr)
    bias = tl.load(bias_ptr) if bias_ptr is not None else 0
    linear_output = tl.dot(input_tensor, weight.T) + bias
    
    # ReLU activation
    relu_output = tl.maximum(linear_output, 0)
    
    # Layer normalization
    mean = tl.sum(relu_output) / normalized_shape
    variance = tl.sum((relu_output - mean) ** 2) / normalized_shape
    std = tl.sqrt(variance + eps)
    
    if elementwise_affine:
        # Apply learnable parameters for layer normalization
        gamma = tl.load(gamma_ptr)
        beta = tl.load(beta_ptr)
        output = gamma * (relu_output - mean) / std + beta
    else:
        output = (relu_output - mean) / std
    
    # Store the result
    tl.store(output_ptr, output)

import torch
import triton
import triton.language as tl

def fused_layer_norm_relu_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, 
                                  normalized_shape: int = None, eps: float = 1e-5, 
                                  elementwise_affine: bool = True) -> torch.Tensor:
    # Ensure input shapes are correct
    assert input.dim() == 2, "Input tensor must be 2D"
    assert weight.dim() == 2, "Weight tensor must be 2D"
    if bias is not None:
        assert bias.dim() == 1, "Bias tensor must be 1D"
    
    # Prepare output tensor
    output = torch.empty(input.size(0), weight.size(0), device=input.device, dtype=input.dtype)
    
    # Launch the Triton kernel
    grid = (input.size(0),)
    fused_layer_norm_relu_linear_kernel[grid](input, weight, bias, output, 
                                               normalized_shape, eps, elementwise_affine, 
                                               n_elements=input.size(1))
    
    return output
