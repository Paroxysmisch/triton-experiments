import torch
import triton
import triton.language as tl

@triton.jit
def fused_mul_add_logsoftmax_dropout_kernel(
    input1_ptr, input2_ptr, other_ptr, output_ptr,
    # Tensor dimensions
    B, N, D,
    # Strides for input1
    input1_batch_stride, input1_row_stride, input1_feature_stride,
    # Strides for input2
    input2_batch_stride, input2_row_stride, input2_feature_stride,
    # Strides for other
    other_batch_stride, other_row_stride, other_feature_stride,
    # Strides for output
    output_batch_stride, output_row_stride, output_feature_stride,
    # Parameters
    p, training, dim,
    BLOCK_SIZE: tl.constexpr,
    LOG2_BLOCK_SIZE: tl.constexpr,
):
    # Compute the row index for each program
    b = tl.program_id(0)
    n = tl.program_id(1)
    
    # Offsets for the current row
    off_b = b
    off_n = n
    
    # Pointers to the start of the current row for each input
    input1_row_ptr = input1_ptr + off_b * input1_batch_stride + off_n * input1_row_stride
    input2_row_ptr = input2_ptr + off_b * input2_batch_stride + off_n * input2_row_stride
    other_row_ptr = other_ptr + off_b * other_batch_stride + off_n * other_row_stride
    
    # Block pointer for features
    feature_range = tl.arange(0, BLOCK_SIZE)
    mask = feature_range < D
    
    # Load inputs with broadcasting support
    i1 = tl.load(input1_row_ptr + feature_range * input1_feature_stride, mask=mask, other=0.0)
    i2 = tl.load(input2_row_ptr + feature_range * input2_feature_stride, mask=mask, other=0.0)
    other = tl.load(other_row_ptr + feature_range * other_feature_stride, mask=mask, other=0.0)
    
    # Compute element-wise operations
    z = i1 * i2
    s = z + other
    
    # Log-softmax computation
    max_s = tl.max(s, axis=0)
    s_centered = s - max_s
    exp_s = tl.exp(s_centered)
    sum_exp = tl.sum(exp_s, axis=0)
    log_sum_exp = tl.log(sum_exp) + max_s
    log_softmax = s - log_sum_exp
    
    # Apply dropout
    if training:
        rand_vals = tl.rand(feature_range, seed=tl.program_id(2))
        mask_dropout = rand_vals > p
        dropout_scale = 1.0 / (1.0 - p)
        log_softmax = log_softmax * tl.where(mask_dropout, dropout_scale, 0.0)
    
    # Store the result
    output_row_ptr = output_ptr + off_b * output_batch_stride + off_n * output_row_stride
    tl.store(output_row_ptr + feature_range * output_feature_stride, log_softmax, mask=mask)

def fused_mul_add_logsoftmax_dropout(input1, input2, other, p=0.5, training=True, inplace=False, dim=-1):
    # Ensure inputs are broadcastable
    assert input1.shape == input2.shape, "input1 and input2 must have the same shape for broadcasting"
    assert torch.broadcast_shapes(input1.shape, other.shape) is not None, "other must be broadcastable to input1's shape"
    
    # Compute the broadcasted shape
    broadcast_shape = torch.broadcast_shapes(input1.shape, input2.shape, other.shape)
    B, N, D = broadcast_shape if len(broadcast_shape) == 3 else (1, *broadcast_shape)
    
    # Allocate output tensor
    if inplace:
        output = input1
        assert input1.is_contiguous(), "inplace operation requires input1 to be contiguous"
    else:
        output = torch.empty_like(input1.expand(broadcast_shape))
    
    # Check if dimensions are compatible
    assert dim == -1 or dim == len(broadcast_shape) - 1, "log_softmax must be applied along the last dimension"
    
    # Launch kernel
    BLOCK_SIZE = triton.next_power_of_2(D)
    if BLOCK_SIZE > 4096:
        BLOCK_SIZE = 4096
    
    grid = (B, N)
    fused_mul_add_logsoftmax_dropout_kernel[grid](
        input1, input2, other, output,
        B, N, D,
        input1.stride(0), input1.stride(1), input1.stride(2),
        input2.stride(0), input2.stride(1), input2.stride(2),
        other.stride(0) if other.dim() >= 1 else 0,
        other.stride(1) if other.dim() >= 2 else 0,
        other.stride(2) if other.dim() >= 3 else 0,
        output.stride(0), output.stride(1), output.stride(2),
        p, training, dim,
        BLOCK_SIZE=BLOCK_SIZE,
        LOG2_BLOCK_SIZE=BLOCK_SIZE.bit_length() - 1,
    )
    
    return output

def fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, p=0.5, training=True, inplace=False, dim=-1, *, out=None):
    # Compute fused operations up to dropout
    d = fused_mul_add_logsoftmax_dropout(input1, input2, other, p, training, inplace, dim)
    
    # Perform batch matrix multiplication
    y = torch.bmm(d, mat2)
    
    # Handle output tensor
    if out is not None:
        out.copy_(y)
        return out
    return y
