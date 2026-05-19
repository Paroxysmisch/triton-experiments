import triton
import triton.language as tl

@triton.jit
def fused_kernel(X1_ptr, X2_ptr, O_ptr, M_ptr, Y_ptr, 
                 B, N, D_in, D_out, 
                 p, dim, training, 
                 X1_stride, X2_stride, O_stride, M_stride, Y_stride):
    # Compute block indices
    batch_id = tl.program_id(0)
    n_id = tl.program_id(1)
    d_in_id = tl.program_id(2)

    # Compute memory offsets
    X1_offset = batch_id * X1_stride[0] + n_id * X1_stride[1] + d_in_id * X1_stride[2]
    X2_offset = batch_id * X2_stride[0] + n_id * X2_stride[1] + d_in_id * X2_stride[2]
    O_offset = batch_id * O_stride[0] + n_id * O_stride[1] + d_in_id * O_stride[2]
    M_offset = batch_id * M_stride[0] + d_in_id * M_stride[1]
    Y_offset = batch_id * Y_stride[0] + n_id * Y_stride[1] + d_in_id * Y_stride[2]

    # Load data
    X1 = tl.load(X1_ptr + X1_offset)
    X2 = tl.load(X2_ptr + X2_offset)
    O = tl.load(O_ptr + O_offset)

    # Element-wise multiplication and addition
    Z = X1 * X2
    S = Z + O

    # Log-softmax
    max_s = tl.max(S, axis=dim)
    S = S - max_s
    exp_s = tl.exp(S)
    sum_exp_s = tl.sum(exp_s, axis=dim)
    L = S - tl.log(sum_exp_s)

    # Dropout
    if training:
        mask = tl.rand(L.shape) > p
        D = L * mask
    else:
        D = L

    # Batch matrix multiplication
    Y = tl.dot(D, tl.load(M_ptr + M_offset))

    # Store result
    tl.store(Y_ptr + Y_offset, Y)

import torch

def fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, p=0.5, training=True, inplace=False, dim=-1, *, out=None):
    # Check input shapes and broadcastability
    if not (input1.shape == input2.shape == other.shape):
        raise ValueError("input1, input2, and other must be broadcastable to each other")

    # Get shapes
    B, N, D_in = input1.shape
    _, _, D_out = mat2.shape

    # Allocate output tensor
    if out is None:
        out = torch.empty((B, N, D_out), device=input1.device, dtype=input1.dtype)

    # Launch Triton kernel
    grid = (B, N, D_in)
    fused_kernel[grid](
        input1, input2, other, mat2, out,
        B, N, D_in, D_out,
        p, dim, training,
        input1.stride(), input2.stride(), other.stride(), mat2.stride(), out.stride()
    )

    return out
