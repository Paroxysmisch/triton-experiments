import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def fused_layer_norm_relu_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr, 
    in_features, out_features, eps, 
    N_FEATURES: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)

    input_idx = row_start + offsets
    mask = input_idx < in_features

    # Load input row
    input = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    
    # Load weight row
    weight = tl.load(weight_ptr + offsets, mask=offsets < out_features)
    
    # Linear transformation
    output = tl.dot(input, weight)
    
    # Add bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offsets, mask=offsets < out_features)
        output += bias
    
    # Apply ReLU
    output = tl.max(output, 0.0)
    
    # Layer normalization
    mean = tl.sum(output, axis=0) / N_FEATURES
    var = tl.sum((output - mean) * (output - mean), axis=0) / N_FEATURES
    norm_output = (output - mean) / tl.sqrt(var + eps)
    
    # Store result
    tl.store(output_ptr + input_idx, norm_output, mask=mask)

@torch.inference_mode()
def fused_layer_norm_relu_linear(input: Tensor, weight: Tensor, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True) -> Tensor:
    """
    Applies a fused operation of linear transformation, ReLU activation, and layer normalization.

    Args:
        input (Tensor): Input tensor with shape (*, in_features).
        weight (Tensor): Weights for the linear transformation, shape (out_features, in_features).
        bias (Tensor, optional): Bias for the linear transformation, shape (out_features).
        normalized_shape (int or list or torch.Size, optional): Shape of the dimensions to normalize.
        eps (float, optional): A value added to the denominator for numerical stability. Default is 1e-5.
        elementwise_affine (bool, optional): If True, layer normalization has learnable parameters. Default is True.

    Returns:
        Tensor: Result after applying the linear transformation, ReLU, and layer normalization.
    """
    assert input.dim() == 2 and weight.dim() == 2, "Input and weight must be 2D tensors"
    assert input.size(1) == weight.size(1), "Incompatible dimensions for input and weight"
    
    in_features = input.size(1)
    out_features = weight.size(0)
    BLOCK_SIZE = triton.next_power_of_2(in_features)
    
    output = torch.empty((input.size(0), out_features), device=input.device, dtype=input.dtype)
    
    grid = (input.size(0),)
    
    fused_layer_norm_relu_linear_kernel[grid](
        input, weight, bias, output, 
        in_features, out_features, eps, 
        in_features, BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
        device=input.device
    )
    
    return output
