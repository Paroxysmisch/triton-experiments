import triton
import triton.language as tl

@triton.jit
def fused_kernel(X1_ptr, X2_ptr, O_ptr, M_ptr, Y_ptr, stride_x1, stride_x2, stride_o, stride_m, stride_y, N, D_in, D_out, p, training, dim):
    # Offsets for each element in the batch
    batch_idx = tl.program_id(0)
    
    # Load input tensors
    X1 = tl.load(X1_ptr + batch_idx * stride_x1)
    X2 = tl.load(X2_ptr + batch_idx * stride_x2)
    O = tl.load(O_ptr + batch_idx * stride_o)
    
    # Element-wise multiplication
    Z = X1 * X2
    
    # Addition
    S = Z + O
    
    # Log-softmax
    max_s = tl.max(S, axis=dim, keepdims=True)
    S = S - max_s
    exp_s = tl.exp(S)
    sum_exp_s = tl.sum(exp_s, axis=dim, keepdims=True)
    L = S - tl.log(sum_exp_s)
    
    # Dropout
    if training:
        mask = tl.rand(L.shape) > p
        D = L * mask / (1 - p)
    else:
        D = L
    
    # Batch matrix multiplication
    Y = tl.dot(D, M_ptr + batch_idx * stride_m)
    
    # Store result
    tl.store(Y_ptr + batch_idx * stride_y, Y)


import torch

def fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, p=0.5, training=True, inplace=False, dim=-1, *, out=None):
    # Ensure inputs are compatible
    assert input1.shape == input2.shape, "input1 and input2 must have the same shape"
    assert other.shape == input1.shape or other.numel() == 1, "other must be broadcastable to input1"
    
    # Get shapes
    B, N, D_in = input1.shape
    _, _, D_out = mat2.shape
    
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty((B, N, D_out), device=input1.device, dtype=input1.dtype)
    
    # Launch Triton kernel
    grid = (B,)
    fused_kernel[grid](
        input1, input2, other, mat2, out,
        input1.stride(0), input2.stride(0), other.stride(0), mat2.stride(0), out.stride(0),
        N, D_in, D_out, p, training, dim
    )
    
    return out
