import triton
import triton.language as tl
import torch

@triton.jit
def linear_sigmoid_dropout_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr, 
    in_features, out_features, p, is_training, 
    BLOCK_SIZE: tl.constexpr
):
    # Position in the output tensor
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load input row
    input_row = tl.load(input_ptr + row_idx * in_features + col_idx, mask=col_idx < in_features, other=0.0)
    
    # Load weight row
    weight_row = tl.load(weight_ptr + col_idx * in_features, mask=col_idx < out_features, other=0.0)
    
    # Compute linear transformation
    output = tl.dot(input_row, weight_row)
    
    # Add bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + col_idx, mask=col_idx < out_features, other=0.0)
        output += bias
    
    # Apply sigmoid activation
    output = 1 / (1 + tl.exp(-output))
    
    # Apply dropout if training
    if is_training:
        keep_prob = 1 - p
        random_tensor = tl.rand(output.shape, dtype=tl.float32)
        dropout_mask = random_tensor < keep_prob
        output = tl.where(dropout_mask, output / keep_prob, 0.0)
    
    # Store the result
    tl.store(output_ptr + row_idx * out_features + col_idx, output, mask=col_idx < out_features)


def dropout_sigmoid_linear(input: torch.Tensor, weight: torch.Tensor, bias=None, p=0.5, training=True, inplace=False) -> torch.Tensor:
    # Check input dimensions
    assert input.dim() >= 2, "Input tensor must have at least 2 dimensions"
    assert weight.dim() == 2, "Weight tensor must be 2D"
    if bias is not None:
        assert bias.dim() == 1, "Bias tensor must be 1D"
    
    # Extract dimensions
    in_features = input.shape[-1]
    out_features = weight.shape[0]
    
    # Prepare output tensor
    output_shape = list(input.shape)
    output_shape[-1] = out_features
    output = input if inplace else torch.empty(*output_shape, device=input.device, dtype=input.dtype)
    
    # Launch Triton kernel
    grid = (input.shape[0], (out_features + 31) // 32)
    linear_sigmoid_dropout_kernel[grid](
        input, weight, bias, output,
        in_features, out_features, p, training,
        BLOCK_SIZE=32
    )
    
    return output
