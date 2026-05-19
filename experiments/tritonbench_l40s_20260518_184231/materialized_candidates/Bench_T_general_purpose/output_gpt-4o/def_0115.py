import triton
import triton.language as tl
import torch

@triton.jit
def tanh_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    in_features, out_features, has_bias,
    BLOCK_SIZE: tl.constexpr
):
    # Get the row and column index of the current block
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Create pointers to the current row and column
    input_offset = row_idx * in_features
    weight_offset = col_idx * in_features
    output_offset = row_idx * out_features + col_idx
    
    # Load input and weight
    input = tl.load(input_ptr + input_offset + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < in_features, other=0.0)
    weight = tl.load(weight_ptr + weight_offset + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < in_features, other=0.0)
    
    # Compute the dot product
    dot_product = tl.dot(input, weight)
    
    # Add bias if available
    if has_bias:
        bias = tl.load(bias_ptr + col_idx)
        dot_product += bias
    
    # Apply Tanh activation
    result = (tl.exp(dot_product) - tl.exp(-dot_product)) / (tl.exp(dot_product) + tl.exp(-dot_product))
    
    # Store the result
    tl.store(output_ptr + output_offset, result)

def tanh_linear(input, weight, bias=None):
    # Ensure input is a 2D tensor
    if input.dim() < 2:
        raise ValueError("Input tensor must have at least 2 dimensions")
    
    # Get the dimensions
    in_features = input.shape[-1]
    out_features = weight.shape[0]
    
    # Prepare output tensor
    output_shape = list(input.shape)
    output_shape[-1] = out_features
    output = torch.empty(output_shape, device=input.device, dtype=input.dtype)
    
    # Determine if bias is used
    has_bias = bias is not None
    
    # Launch the Triton kernel
    grid = (output.shape[0], out_features)
    tanh_linear_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias if has_bias else torch.tensor(0, device=input.device),
        output_ptr=output,
        in_features=in_features,
        out_features=out_features,
        has_bias=has_bias,
        BLOCK_SIZE=128
    )
    
    return output
