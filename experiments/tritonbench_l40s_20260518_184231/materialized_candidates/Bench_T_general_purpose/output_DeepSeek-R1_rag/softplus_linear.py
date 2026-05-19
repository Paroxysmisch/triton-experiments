import torch
import triton
import triton.language as tl

@triton.jit
def softplus_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_row_stride, input_col_stride,
    weight_row_stride, weight_col_stride,
    bias_row_stride, bias_col_stride,
    output_row_stride, output_col_stride,
    n_rows, n_cols,
    beta,
    threshold,
    BLOCK_SIZE_ROW: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr,
    HAS_BIAS: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    if row_idx >= n_rows or col_idx >= n_cols:
        return
    
    input_offset = row_idx * input_row_stride + col_idx * input_col_stride
    weight_offset = row_idx * weight_row_stride + col_idx * weight_col_stride
    if HAS_BIAS:
        bias_offset = row_idx * bias_row_stride + col_idx * bias_col_stride
    
    input_val = tl.load(input_ptr + input_offset)
    weight_val = tl.load(weight_ptr + weight_offset)
    bias_val = tl.load(bias_ptr + bias_offset) if HAS_BIAS else 0.0
    
    linear = input_val * weight_val + bias_val
    bx = beta * linear
    exp_bx = tl.exp(bx)
    softplus_val = tl.log(1.0 + exp_bx) / beta
    output_val = tl.where(bx > threshold, linear, softplus_val)
    
    output_offset = row_idx * output_row_stride + col_idx * output_col_stride
    tl.store(output_ptr + output_offset, output_val)

def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
    assert input.dim() == weight.dim(), "Input and weight must have the same number of dimensions"
    try:
        broadcast_shape = torch.broadcast_shapes(input.shape, weight.shape)
    except Exception as e:
        raise ValueError("Input and weight are not broadcastable") from e
    
    if bias is not None:
        try:
            torch.broadcast_shapes(broadcast_shape, bias.shape)
        except Exception as e:
            raise ValueError("Bias is not broadcastable with input and weight") from e
    
    expanded_input = input.expand(broadcast_shape)
    expanded_weight = weight.expand(broadcast_shape)
    expanded_bias = bias.expand(broadcast_shape) if bias is not None else None
    
    output = torch.empty_like(expanded_input)
    n_rows, n_cols = broadcast_shape[-2], broadcast_shape[-1]
    
    def stride_last_two(t):
        if t.dim() >= 2:
            return t.stride()[-2], t.stride()[-1]
        return (0, 0)
    
    input_row_stride, input_col_stride = stride_last_two(expanded_input)
    weight_row_stride, weight_col_stride = stride_last_two(expanded_weight)
    if expanded_bias is not None:
        bias_row_stride, bias_col_stride = stride_last_two(expanded_bias)
    else:
        bias_row_stride, bias_col_stride = 0, 0
    
    output_row_stride, output_col_stride = stride_last_two(output)
    
    grid = (triton.cdiv(n_rows, 1), triton.cdiv(n_cols, 1))
    HAS_BIAS = bias is not None
    
    softplus_linear_kernel[grid](
        expanded_input.data_ptr(),
        expanded_weight.data_ptr(),
        expanded_bias.data_ptr() if HAS_BIAS else 0,
        output.data_ptr(),
        input_row_stride, input_col_stride,
        weight_row_stride, weight_col_stride,
        bias_row_stride, bias_col_stride,
        output_row_stride, output_col_stride,
        n_rows, n_cols,
        beta,
        threshold,
        1, 1,
        HAS_BIAS
    )
    return output
