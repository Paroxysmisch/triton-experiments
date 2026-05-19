import triton
import triton.language as tl

@triton.jit
def fused_mul_add_logsoftmax_dropout_bmm_kernel(
    input1_ptr, input2_ptr, other_ptr, mat2_ptr, output_ptr,
    input1_shape0, input1_shape1, input1_shape2, input1_stride0, input1_stride1, input1_stride2,
    input2_shape0, input2_shape1, input2_shape2, input2_stride0, input2_stride1, input2_stride2,
    other_shape0, other_shape1, other_shape2, other_stride0, other_stride1, other_stride2,
    mat2_shape0, mat2_shape1, mat2_shape2, mat2_stride0, mat2_stride1, mat2_stride2,
    output_shape0, output_shape1, output_shape2, output_stride0, output_stride1, output_stride2,
    dropout_prob: tl.float32, dim: tl.int32, BLOCK_SIZE: tl.constexpr
):
    # Get the batch and feature indices
    pid = tl.program_id(axis=0)
    batch_idx = pid // (input1_shape1 // BLOCK_SIZE)
    feature_idx = pid % (input1_shape1 // BLOCK_SIZE)
    
    # Compute the range of elements to process
    range_start = feature_idx * BLOCK_SIZE
    range_end = min(range_start + BLOCK_SIZE, input1_shape1)
    
    # Initialize the output block
    output_block = tl.zeros((BLOCK_SIZE, mat2_shape2), dtype=tl.float32)
    
    # Load the input1, input2, and other blocks
    input1_block = tl.load(input1_ptr + batch_idx * input1_stride0 + range_start * input1_stride1, mask=range_start + tl.arange(0, BLOCK_SIZE) < input1_shape1)
    input2_block = tl.load(input2_ptr + batch_idx * input2_stride0 + range_start * input2_stride1, mask=range_start + tl.arange(0, BLOCK_SIZE) < input2_shape1)
    other_block = tl.load(other_ptr + batch_idx * other_stride0 + range_start * other_stride1, mask=range_start + tl.arange(0, BLOCK_SIZE) < other_shape1)
    
    # Perform element-wise multiplication and addition
    z_block = input1_block * input2_block
    s_block = z_block + other_block
    
    # Apply log-softmax
    s_block = s_block.to(tl.float32)
    max_val = tl.max(s_block, axis=1)[:, None]
    exp_s = tl.exp(s_block - max_val)
    sum_exp_s = tl.sum(exp_s, axis=1)[:, None]
    log_softmax_block = s_block - max_val - tl.log(sum_exp_s)
    
    # Apply dropout
    if dropout_prob > 0.0 and dropout_prob < 1.0:
        mask = tl.rand(tl.float32, (BLOCK_SIZE,)) < (1.0 - dropout_prob)
        log_softmax_block = tl.where(mask, log_softmax_block / (1.0 - dropout_prob), 0.0)
    
    # Perform batch matrix multiplication
    mat2_block = tl.load(mat2_ptr + batch_idx * mat2_stride0 + range_start * mat2_stride1, mask=range_start + tl.arange(0, BLOCK_SIZE) < mat2_shape1)
    output_block = tl.dot(log_softmax_block, mat2_block)
    
    # Store the result
    tl.store(output_ptr + batch_idx * output_stride0 + range_start * output_stride1, output_block, mask=range_start + tl.arange(0, BLOCK_SIZE) < output_shape1)

import torch
import triton
import triton.language as tl

def fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, p=0.5, training=True, inplace=False, dim=-1, *, out=None):
    # Ensure input tensors are on the same device
    device = input1.device
    input1 = input1.to(device)
    input2 = input2.to(device)
    other = other.to(device)
    mat2 = mat2.to(device)
    
    # Ensure output tensor is on the same device
    if out is not None:
        out = out.to(device)
    
    # Get the shapes and strides
    input1_shape = input1.shape
    input2_shape = input2.shape
    other_shape = other.shape
    mat2_shape = mat2.shape
    output_shape = (input1_shape[0], input1_shape[1], mat2_shape[2])
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty(output_shape, dtype=input1.dtype, device=device)
    
    # Define the grid and block sizes
    grid = (input1_shape[0] * (input1_shape[1] // 128),)
    block = (128,)
    
    # Launch the Triton kernel
    fused_mul_add_logsoftmax_dropout_bmm_kernel[grid, block](
        input1, input2, other, mat2, out,
        *input1_shape, *input1.stride(),
        *input2_shape, *input2.stride(),
        *other_shape, *other.stride(),
        *mat2_shape, *mat2.stride(),
        *output_shape, *out.stride(),
        p, dim
    )
    
    return out
