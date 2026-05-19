import torch
import triton
import triton.language as tl
import math

@triton.jit
def fused_mul_add_logsoftmax_dropout_bmm_kernel(
    # Pointers to matrices
    input1_ptr, input2_ptr, other_ptr, mat2_ptr, output_ptr,
    # Matrix dimensions
    batch_size, seq_len, feat_dim, out_dim,
    # Other parameters
    p, seed, training,
    # Strides
    stride_input1_b, stride_input1_s, stride_input1_f,
    stride_input2_b, stride_input2_s, stride_input2_f,
    stride_other_b, stride_other_s, stride_other_f,
    stride_mat2_b, stride_mat2_f, stride_mat2_o,
    stride_output_b, stride_output_s, stride_output_o,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Compute batch and position indices
    pid = tl.program_id(0)
    batch_idx = pid // (seq_len * out_dim)
    seq_pos = (pid % (seq_len * out_dim)) // out_dim
    out_pos = pid % out_dim

    # Initialize pointers for current batch
    input1_batch_ptr = input1_ptr + batch_idx * stride_input1_b
    input2_batch_ptr = input2_ptr + batch_idx * stride_input2_b
    other_batch_ptr = other_ptr + batch_idx * stride_other_b
    mat2_batch_ptr = mat2_ptr + batch_idx * stride_mat2_b
    
    # Load and compute element-wise multiplication
    offs_f = tl.arange(0, BLOCK_SIZE_K)
    mask_f = offs_f < feat_dim
    
    input1 = tl.load(input1_batch_ptr + seq_pos * stride_input1_s + offs_f * stride_input1_f, mask=mask_f)
    input2 = tl.load(input2_batch_ptr + seq_pos * stride_input2_s + offs_f * stride_input2_f, mask=mask_f)
    other = tl.load(other_batch_ptr + seq_pos * stride_other_s + offs_f * stride_other_f, mask=mask_f)
    
    # Element-wise operations
    mul = input1 * input2
    add = mul + other
    
    # Log-softmax computation
    max_val = tl.max(add, axis=0)
    exp_val = tl.exp(add - max_val)
    sum_exp = tl.sum(exp_val, axis=0)
    log_softmax = add - max_val - tl.log(sum_exp)
    
    # Dropout
    if training:
        rand = tl.rand(seed, offs_f)
        dropout_mask = rand > p
        log_softmax = tl.where(dropout_mask, log_softmax / (1.0 - p), 0.0)
    
    # Batch matrix multiplication
    acc = 0.0
    for k in range(0, feat_dim, BLOCK_SIZE_K):
        k_offs = k + offs_f
        k_mask = k_offs < feat_dim
        a = log_softmax
        b = tl.load(mat2_batch_ptr + k_offs * stride_mat2_f + out_pos * stride_mat2_o, mask=k_mask)
        acc += tl.sum(a * b, axis=0)
    
    # Store result
    output_ptr_idx = batch_idx * stride_output_b + seq_pos * stride_output_s + out_pos * stride_output_o
    tl.store(output_ptr + output_ptr_idx, acc)

def fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, p=0.5, training=True, inplace=False, dim=-1, *, out=None):
    # Input validation
    assert input1.dim() == 3 and input2.dim() == 3, "input1 and input2 must be 3D tensors"
    assert mat2.dim() == 3, "mat2 must be a 3D tensor"
    
    batch_size, seq_len, feat_dim = input1.shape
    _, _, out_dim = mat2.shape
    
    # Ensure shapes are compatible
    assert input1.shape == input2.shape, "input1 and input2 must have the same shape"
    assert mat2.shape[0] == batch_size and mat2.shape[1] == feat_dim, "mat2 has incompatible dimensions"
    
    # Handle output tensor
    if out is None:
        out = torch.empty((batch_size, seq_len, out_dim), device=input1.device, dtype=input1.dtype)
    elif not inplace:
        out = out.clone()
    
    # Prepare grid and block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    grid = (batch_size * seq_len * out_dim,)
    
    # Launch kernel
    fused_mul_add_logsoftmax_dropout_bmm_kernel[grid](
        input1, input2, other, mat2, out,
        batch_size, seq_len, feat_dim, out_dim,
        p, torch.randint(0, 2**31-1, (1,)).item(), training,
        input1.stride(0), input1.stride(1), input1.stride(2),
        input2.stride(0), input2.stride(1), input2.stride(2),
        other.stride(0), other.stride(1), other.stride(2),
        mat2.stride(0), mat2.stride(1), mat2.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return out
