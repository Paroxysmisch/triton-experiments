triton
import triton
import triton.language as tl

@triton.jit
def fused_masked_select_add_gelu_kernel(
    X_ptr,  # Input tensor
    M_ptr,  # Mask tensor
    O_ptr,  # Other tensor or scalar
    Y_ptr,  # Output tensor
    N: int,   # Number of elements in the input tensor
    stride_X: int,  # Stride of the input tensor
    stride_M: int,  # Stride of the mask tensor
    stride_O: int,  # Stride of the other tensor
    stride_Y: int,  # Stride of the output tensor
    alpha: float,  # Scaling factor for the other tensor
    approximate: str,  # Approximation method for GELU
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    x_indices = block_start + offsets
    valid_mask = x_indices < N

    # Load input, mask, and other tensors
    X = tl.load(X_ptr + x_indices[:, None] * stride_X, mask=valid_mask)
    M = tl.load(M_ptr + x_indices[:, None] * stride_M, mask=valid_mask)
    O = tl.load(O_ptr + x_indices[:, None] * stride_O, mask=valid_mask)

    # Apply masked select
    Z = X * M

    # Add scaled other tensor
    S = Z + alpha * O

    # Apply GELU activation
    if approximate == 'tanh':
        # Approximate GELU using tanh
        gelu_approx = 0.5 * (1 + tl.tanh(tl.sqrt(2 / tl.pi) * (S + 0.044715 * S ** 3)))
    else:
        # Exact GELU
        gelu_exact = 0.5 * (S + tl.sqrt(2 / tl.pi) * S * tl.exp(-0.5 * S ** 2))

    # Store the result
    Y = gelu_approx if approximate == 'tanh' else gelu_exact
    tl.store(Y_ptr + x_indices[:, None] * stride_Y, Y, mask=valid_mask)


BLOCK_SIZE = 1024


def fused_masked_select_add_gelu(x, m, o, *, alpha=1, approximate='none'):
    assert isinstance(x, triton.Tensor)
    assert isinstance(m, triton.Tensor)
    assert isinstance(o, triton.Tensor)
    
    # Get shapes and strides
    n = x.shape[0]
    stride_x = x.stride[0]
    stride_m = m.stride[0]
    stride_o = o.stride[0]
    
    # Allocate output tensor
    y = triton.Tensor(shape=(n,), dtype=x.dtype)
    stride_y = y.stride[0]
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(n, BLOCK_SIZE),)
    fused_masked_select_add_gelu_kernel[grid](
        x.data_ptr(), m.data_ptr(), o.data_ptr(), y.data_ptr(),
        n, stride_x, stride_m, stride_o, stride_y,
        alpha, approximate
    )
    
    return y
