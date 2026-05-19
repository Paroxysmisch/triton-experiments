import triton
import triton.language as tl

@triton.jit
def fused_mul_add_logsoftmax_dropout_bmm_kernel(
    X1_ptr, X2_ptr, O_ptr, mat2_ptr, Y_ptr,
    X1_shape, X2_shape, O_shape, mat2_shape, Y_shape,
    p, training, inplace, dim, block_size=256):
    
    # Extract dimensions
    B = X1_shape[0]
    N = X1_shape[1]
    D_in = X1_shape[2]
    D_out = mat2_shape[2]
    
    # Initialize pointers
    x1_idx = tl.program_id(axis=0)
    n_idx = tl.program_id(axis=1)
    d_idx = tl.program_id(axis=2)
    
    # Element-wise multiplication
    z = X1_ptr[x1_idx * N * D_in + n_idx * D_in + d_idx] * \
        X2_ptr[x1_idx * N * D_in + n_idx * D_in + d_idx]
    
    # Addition
    s = z + O_ptr[n_idx * D_in + d_idx] if O_shape == (N, D_in) else O_ptr
    
    # Log-Softmax
    exp_s = tl.exp(s)
    sum_exp_s = tl.sum(exp_s, axis=dim)
    log_sum_exp_s = tl.log(sum_exp_s)
    l = s - log_sum_exp_s
    
    # Dropout
    mask = tl.random.dropout_mask(l.shape, p, training)
    d = l * mask
    
    # Batch Matrix Multiplication
    y = tl.dot(d, mat2_ptr[:, :, d_idx:D_idx+1], trans_a=False, trans_b=True)
    
    # Store result
    if inplace:
        Y_ptr[x1_idx * N * D_out + n_idx * D_out + d_idx] = y
    else:
        Y_ptr[x1_idx * N * D_out + n_idx * D_out + d_idx] = y

# Wrapper function
def fused_mul_add_logsoftmax_dropout_bmm(input1, input2, other, mat2, p=0.5, training=True, inplace=False, dim=-1, *, out=None):
    # Shapes
    X1_shape = input1.shape
    X2_shape = input2.shape
    O_shape = other.shape
    mat2_shape = mat2.shape
    Y_shape = (X1_shape[0], X1_shape[1], mat2_shape[2])
    
    # Allocate memory for output
    if out is None:
        out = tl.zeros(Y_shape, dtype=input1.dtype)
    else:
        assert out.shape == Y_shape and out.dtype == input1.dtype
    
    # Launch kernel
    grid = (X1_shape[0], X1_shape[1], X1_shape[2])
    fused_mul_add_logsoftmax_dropout_bmm_kernel[X1_shape[0], X1_shape[1], X1_shape[2]](
        input1.data, input2.data, other.data, mat2.data, out.data,
        X1_shape, X2_shape, O_shape, mat2_shape, Y_shape,
        p, training, inplace, dim
    )
    
    return out
