import triton
import triton.language as tl

@triton.jit
def _softmax_log_kernel(
    input_ptr, output_ptr,
    stride_input, stride_output,
    N, C, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE
    row_end = min(row_start + BLOCK_SIZE, N)
    
    # Load the slice of the input tensor
    input_slice = tl.load(input_ptr + row_start * stride_input, mask=row_end - row_start > 0)
    
    # Compute the log of each element
    log_input = tl.math.log(input_slice)
    
    # Find the maximum value for numerical stability
    max_val = tl.max(log_input, axis=0)
    
    # Subtract the maximum value from each element
    shifted_log_input = log_input - max_val
    
    # Exponentiate the shifted log inputs
    exp_values = tl.math.exp(shifted_log_input)
    
    # Sum the exponentiated values
    sum_exp = tl.sum(exp_values, axis=0)
    
    # Compute the log of the sum of exponentials
    log_sum_exp = tl.math.log(sum_exp) + max_val
    
    # Normalize the exponential values
    normalized_values = exp_values / sum_exp
    
    # Store the result in the output tensor
    tl.store(output_ptr + row_start * stride_output, normalized_values, mask=row_end - row_start > 0)
