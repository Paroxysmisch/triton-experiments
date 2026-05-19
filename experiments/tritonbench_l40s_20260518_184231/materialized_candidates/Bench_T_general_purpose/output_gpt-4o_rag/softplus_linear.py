import torch
import triton
import triton.language as tl

@triton.jit
def softplus_linear_kernel(
    output_ptr, input_ptr, weight_ptr, bias_ptr, beta, threshold, 
    input_row_stride, weight_row_stride, weight_col_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=0.0)

    # Perform linear transformation
    weight_ptrs = weight_ptr + col_offsets * weight_col_stride
    weight = tl.load(weight_ptrs, mask=col_offsets < n_cols, other=0.0)
    linear_output = tl.dot(row, weight)
    
    if bias_ptr:
        bias = tl.load(bias_ptr + col_offsets, mask=col_offsets < n_cols, other=0.0)
        linear_output += bias

    # Apply Softplus activation
    softplus_output = (1 / beta) * tl.log(1 + tl.exp(beta * linear_output))
    
    # For numerical stability, approximate with linear function for large values
    mask_large = linear_output > threshold
    softplus_output = tl.where(mask_large, linear_output, softplus_output)

    # Write back to output
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softplus_output, mask=col_offsets < n_cols)

def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
    assert input.shape[1] == weight.shape[0], "Incompatible input and weight shapes"
    n_rows, n_cols = input.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    output = torch.empty((n_rows, weight.shape[1]), device=input.device, dtype=input.dtype)
    
    softplus_linear_kernel[(n_rows,)](
        output, input, weight, bias, beta, threshold,
        input.stride(0), weight.stride(0), weight.stride(1), output.stride(0), n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output
