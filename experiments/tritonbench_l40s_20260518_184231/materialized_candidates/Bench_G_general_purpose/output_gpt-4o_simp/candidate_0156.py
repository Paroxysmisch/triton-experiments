import triton
import triton.language as tl

# Triton kernel for softmax
@triton.jit
def _softmax(Y, X, M, stride_z, stride_y, stride_x, stride_m, n_cols, log, mask_type, causal, BLOCK_SIZE: tl.constexpr):
    # Get the block index
    z = tl.program_id(0)
    y = tl.program_id(1)

    # Compute the offset for this block
    offset = z * stride_z + y * stride_y

    # Load the input block
    x_ptrs = X + offset + tl.arange(0, BLOCK_SIZE) * stride_x
    x = tl.load(x_ptrs, mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=-float('inf'))

    # Apply mask if provided
    if mask_type != 'none':
        m_ptrs = M + offset + tl.arange(0, BLOCK_SIZE) * stride_m
        mask = tl.load(m_ptrs, mask=tl.arange(0, BLOCK_SIZE) < n_cols)
        x = tl.where(mask, x, -float('inf'))

    # Causal mask
    if causal:
        col_idx = tl.arange(0, BLOCK_SIZE)
        x = tl.where(col_idx[:, None] >= col_idx[None, :], x, -float('inf'))

    # Compute softmax
    max_x = tl.max(x, axis=0)
    x_exp = tl.exp(x - max_x)
    sum_x_exp = tl.sum(x_exp, axis=0)
    y = x_exp / sum_x_exp

    # Apply log if needed
    if log:
        y = tl.log(y)

    # Store the result
    y_ptrs = Y + offset + tl.arange(0, BLOCK_SIZE) * stride_y
    tl.store(y_ptrs, y, mask=tl.arange(0, BLOCK_SIZE) < n_cols)

# Triton kernel for softmax backward
@triton.jit
def _softmax_backward(grad_input, grad_output, softmax_output, stride_z, stride_y, stride_x, n_cols, BLOCK_SIZE: tl.constexpr):
    # Get the block index
    z = tl.program_id(0)
    y = tl.program_id(1)

    # Compute the offset for this block
    offset = z * stride_z + y * stride_y

    # Load the grad_output and softmax_output block
    grad_out_ptrs = grad_output + offset + tl.arange(0, BLOCK_SIZE) * stride_x
    grad_out = tl.load(grad_out_ptrs, mask=tl.arange(0, BLOCK_SIZE) < n_cols)

    softmax_out_ptrs = softmax_output + offset + tl.arange(0, BLOCK_SIZE) * stride_x
    softmax_out = tl.load(softmax_out_ptrs, mask=tl.arange(0, BLOCK_SIZE) < n_cols)

    # Compute gradient of softmax
    dot_product = tl.sum(grad_out * softmax_out, axis=0)
    grad_in = softmax_out * (grad_out - dot_product)

    # Store the result
    grad_in_ptrs = grad_input + offset + tl.arange(0, BLOCK_SIZE) * stride_y
    tl.store(grad_in_ptrs, grad_in, mask=tl.arange(0, BLOCK_SIZE) < n_cols)

# Wrapper function for softmax
def softmax(Y, X, M=None, log=False, mask_type='none', causal=False):
    # Determine the shape of the input tensor
    Z, Y_dim, X_dim = X.shape

    # Define block size and launch grid
    BLOCK_SIZE = 128
    grid = (Z, Y_dim)

    # Launch the Triton kernel
    _softmax[grid](Y, X, M, X.stride(0), X.stride(1), X.stride(2), M.stride(2) if M is not None else 0, X_dim, log, mask_type, causal, BLOCK_SIZE)

# Wrapper function for softmax backward
def softmax_backward(grad_input, grad_output, softmax_output):
    # Determine the shape of the input tensor
    Z, Y_dim, X_dim = softmax_output.shape

    # Define block size and launch grid
    BLOCK_SIZE = 128
    grid = (Z, Y_dim)

    # Launch the Triton kernel
    _softmax_backward[grid](grad_input, grad_output, softmax_output, softmax_output.stride(0), softmax_output.stride(1), softmax_output.stride(2), X_dim, BLOCK_SIZE)
