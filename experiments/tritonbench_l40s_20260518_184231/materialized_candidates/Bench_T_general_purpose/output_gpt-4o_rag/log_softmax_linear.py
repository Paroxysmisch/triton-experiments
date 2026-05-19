import torch
import triton
import triton.language as tl

@triton.jit
def log_softmax_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    in_features, out_features, dim, has_bias: tl.constexpr
):
    # Program ID for parallel execution
    pid = tl.program_id(0)
    
    # Calculate the row and column indices for the output
    row_idx = pid // out_features
    col_idx = pid % out_features
    
    # Load input and weight data
    input_offset = row_idx * in_features
    weight_offset = col_idx * in_features
    input_data = tl.load(input_ptr + input_offset)
    weight_data = tl.load(weight_ptr + weight_offset)
    
    # Compute the linear transformation: y = x * A^T
    y = tl.dot(input_data, weight_data)
    
    # Add bias if it exists
    if has_bias:
        bias_data = tl.load(bias_ptr + col_idx)
        y += bias_data
    
    # Apply log-softmax
    max_y = tl.max(y, 0)
    y_exp = tl.exp(y - max_y)
    sum_exp = tl.sum(y_exp, 0)
    log_softmax_result = tl.log(y_exp / sum_exp)
    
    # Store the result
    tl.store(output_ptr + pid, log_softmax_result)

def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
    # Ensure input is a 2D tensor
    input_shape = input.shape
    input = input.view(-1, input_shape[-1])
    
    # Cast input to desired dtype if specified
    if dtype is not None:
        input = input.to(dtype)
    
    # Get dimensions
    in_features = input.shape[-1]
    out_features = weight.shape[0]
    
    # Prepare output tensor
    output = torch.empty((input.shape[0], out_features), dtype=input.dtype, device=input.device)
    
    # Determine if bias is used
    has_bias = bias is not None
    
    # Launch Triton kernel
    grid = (input.shape[0] * out_features,)
    log_softmax_linear_kernel[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias if has_bias else tl.nullptr,
        output_ptr=output,
        in_features=in_features,
        out_features=out_features,
        dim=dim,
        has_bias=has_bias
    )
    
    # Reshape output to match the input's batch dimensions
    output = output.view(*input_shape[:-1], out_features)
    
    return output
