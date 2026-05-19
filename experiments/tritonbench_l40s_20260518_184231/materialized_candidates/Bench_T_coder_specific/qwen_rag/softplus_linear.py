import triton
import triton.language as tl

@triton.jit
def softplus_linear_kernel(
    output_ptr, 
    input_ptr, 
    weight_ptr, 
    bias_ptr, 
    input_row_stride, 
    input_col_stride, 
    weight_row_stride, 
    weight_col_stride, 
    output_row_stride, 
    output_col_stride, 
    n_rows, 
    n_cols, 
    n_out, 
    beta, 
    threshold, 
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    # Load input and weight data
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_start_ptr = weight_ptr + col_idx * weight_col_stride
    
    input_value = tl.load(row_start_ptr + col_idx * input_col_stride, mask=col_idx < n_cols, other=0)
    weight_value = tl.load(col_start_ptr + row_idx * weight_row_stride, mask=row_idx < n_out, other=0)
    
    # Perform linear transformation
    linear_result = input_value * weight_value
    
    # Apply bias if provided
    if bias_ptr is not None:
        bias_value = tl.load(bias_ptr + row_idx * output_col_stride, mask=row_idx < n_out, other=0)
        linear_result += bias_value
    
    # Compute Softplus
    softplus_result = (1 / beta) * tl.log(1 + tl.exp(beta * linear_result))
    
    # Apply threshold to maintain numerical stability
    softplus_result = tl.where(linear_result > threshold, linear_result, softplus_result)
    
    # Store the result
    output_ptr[row_idx * output_row_stride + col_idx] = softplus_result

def softplus_linear(
    input, 
    weight, 
    bias=None, 
    beta=1, 
    threshold=20
):
    n_rows, n_cols = input.shape
    n_out = weight.shape[0]
    
    # Determine block size
    BLOCK_SIZE = min(1024, n_cols)
    
    # Allocate output tensor
    output = torch.empty((n_rows, n_out), dtype=input.dtype, device=input.device)
    
    # Launch kernel
    grid = (tl.numel(output) // BLOCK_SIZE, 1)
    softmax_linear_kernel[
        grid, 
        BLOCK_SIZE
    ](
        output.data_ptr(), 
        input.data_ptr(), 
        weight.data_ptr(), 
        bias.data_ptr() if bias is not None else None, 
        input.stride(0), 
        input.stride(1), 
        weight.stride(0), 
        weight.stride(1), 
        output.stride(0), 
        output.stride(1), 
        n_rows, 
        n_cols, 
        n_out, 
        beta, 
        threshold, 
        BLOCK_SIZE
    )
    
    return output
