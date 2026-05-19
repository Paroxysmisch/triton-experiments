import torch
import triton
import triton.language as tl

# Kernel: Matrix Multiplication
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Each program handles a full block row of A and block col of B
    row_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offs = tl.arange(0, BLOCK_SIZE)
    # Create pointers
    a_ptrs = A_ptr + (row_offs[:, None] * stride_am + col_offs[None, :] * stride_ak)
    b_ptrs = B_ptr + (col_offs[:, None] * stride_bk)
    # Accumulator
    c = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE):
        a = tl.load(a_ptrs, mask=(row_offs[:, None] < M) & (k + col_offs[None, :] < K), other=0.)
        b = tl.load(b_ptrs, mask=(k + row_offs[:, None] < K) & (col_offs[None, :] < N), other=0.)
        c += tl.dot(a, b)
        # Advance to next K block
        a_ptrs += BLOCK_SIZE * stride_ak
        b_ptrs += BLOCK_SIZE * stride_bk
    # Write output
    c_ptrs = C_ptr + (row_offs[:, None] * stride_cm + tl.arange(0, BLOCK_SIZE)[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=(row_offs[:, None] < M) & (tl.arange(0, BLOCK_SIZE)[None, :] < N))

def triton_matmul(a, b):
    """
    Performs C = a x b using Triton.
    a: [*, M, K]
    b: [K, N]
    returns c: [*, M, N]
    """
    a_shape = a.shape
    b_shape = b.shape

    # We treat the leading dimensions of "a" as batch dims; b has shape [K, N].
    # We'll flatten them for a single matmul per batch, then reshape.
    batch_dims = a_shape[:-2]
    M, K1 = a_shape[-2], a_shape[-1]
    K2, N = b_shape[0], b_shape[1]
    assert K1 == K2, "Incompatible dimensions for matmul."

    a_reshaped = a.view(-1, M, K1)
    batch_size = a_reshaped.shape[0]

    # Allocate output
    c = torch.empty((batch_size, M, N), dtype=torch.float32, device=a.device)

    # Launch grid
    BLOCK_SIZE = 32
    grid = (batch_size,)

    # Strides
    stride_am = a_reshaped.stride(1)
    stride_ak = a_reshaped.stride(2)
    stride_bk = b.stride(0)
    stride_bn = b.stride(1)
    stride_cm = c.stride(1)
    stride_cn = c.stride(2)

    # Launch kernel for each batch
    _matmul_kernel[grid](
        a_reshaped, b, c,
        M, N, K1,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE=BLOCK_SIZE
    )
    # Reshape back
    new_shape = batch_dims + (M, N)
    c = c.view(*new_shape)
    return c

# Kernel: Softmax (applied along last dim)
@triton.jit
def _softmax_kernel(
    X_ptr, Y_ptr,
    stride_xbd, stride_xd,
    stride_ybd, stride_yd,
    B, N,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row
    row_id = tl.program_id(0)
    # Range of row, col
    col_range = tl.arange(0, BLOCK_SIZE)
    row = row_id
    offs = row * stride_xbd
    x_ptrs = X_ptr + offs + col_range * stride_xd
    # Load
    mask = col_range < N
    x = tl.where(mask, tl.load(x_ptrs), float("-inf"))
    # Max
    x_max = tl.max(x, 0)
    x_exp = tl.exp(x - x_max)
    # Sum
    x_sum = tl.sum(x_exp, 0)
    # Normalize
    y = x_exp / x_sum
    # Store
    y_ptrs = Y_ptr + row * stride_ybd + col_range * stride_yd
    tl.store(y_ptrs, tl.where(mask, y, 0.0))

def triton_softmax(x):
    """
    Applies softmax along the last dimension using Triton.
    x: [*, B, N]
    returns: same shape
    """
    shape = x.shape
    B, N = shape[-2], shape[-1]
    leading_dims = shape[:-2]
    x_reshaped = x.view(-1, B, N)
    batch_size = x_reshaped.shape[0]

    y = torch.empty_like(x_reshaped, device=x.device, dtype=x.dtype)

    grid = (batch_size,)

    stride_xbd = x_reshaped.stride(0) * x_reshaped.element_size()
    stride_xd = x_reshaped.stride(2) * x_reshaped.element_size()
    stride_ybd = y.stride(0) * y.element_size()
    stride_yd = y.stride(2) * y.element_size()

    # Convert strides to elements (rather than bytes) if needed
    _softmax_kernel[grid](
        x_reshaped, y,
        stride_xbd, stride_xd,
        stride_ybd, stride_yd,
        B, N,
        BLOCK_SIZE=256
    )

    return y.view(*leading_dims, B, N)

# Dropout (simple Python-based mask; can be fused with Triton but here for clarity)
def dropout(x, p=0.1, training=True):
    if not training or p == 0.0:
        return x
    mask = (torch.rand_like(x) >= p).to(x.dtype)
    return x * mask / (1.0 - p)

# Kernel: Layer Normalization over last dimension with gamma=1, beta=0
@triton.jit
def _layer_norm_kernel(
    X_ptr, Y_ptr,
    stride_xb, stride_xn,
    stride_yb, stride_yn,
    B, N,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    col_range = tl.arange(0, BLOCK_SIZE)
    x_row_ptr = X_ptr + row_id * stride_xb
    y_row_ptr = Y_ptr + row_id * stride_yb
    x = tl.load(x_row_ptr + col_range * stride_xn, mask=(col_range < N), other=0.0)
    mean = tl.sum(x, 0) / N
    var = tl.sum((x - mean) * (x - mean), 0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * rstd
    # gamma=1, beta=0
    out = x_norm
    tl.store(y_row_ptr + col_range * stride_yn, tl.where(col_range < N, out, 0.0))

def triton_layer_norm(x, eps=1e-5):
    """
    Layer normalization across last dimension with gamma=1, beta=0.
    x: [*, B, N]
    returns: same shape
    """
    shape = x.shape
    B, N = shape[-2], shape[-1]
    leading_dims = shape[:-2]
    x_reshaped = x.view(-1, B, N)
    batch_size = x_reshaped.shape[0]

    y = torch.empty_like(x_reshaped, device=x.device, dtype=x.dtype)

    stride_xb = x_reshaped.stride(0) * x_reshaped.element_size()
    stride_xn = x_reshaped.stride(2) * x_reshaped.element_size()
    stride_yb = y.stride(0) * y.element_size()
    stride_yn = y.stride(2) * y.element_size()

    grid = (batch_size,)

    _layer_norm_kernel[grid](
        x_reshaped, y,
        stride_xb, stride_xn,
        stride_yb, stride_yn,
        B, N,
        eps,
        BLOCK_SIZE=256
    )
    return y.view(*leading_dims, B, N)

def fused_transformer_block(input, weight1, weight2, residual, dropout_p=0.1, eps=1e-5, *, out=None, training=True):
    """
    fused_transformer_block(input, weight1, weight2, residual, dropout_p=0.1, eps=1e-5, *, out=None) -> Tensor

    Performs:
      Z1 = input @ weight1
      Z2 = softmax(Z1)
      Z3 = dropout(Z2, p=dropout_p)
      Z4 = Z3 @ weight2
      Y  = LayerNorm(Z4 + residual, eps=eps)

    input:    [*, N, D_in]
    weight1:  [D_in, D_k]
    weight2:  [D_k, D_out]
    residual: broadcastable to [*, N, D_out]
    dropout_p: float, default=0.1
    eps:       float, default=1e-5
    out:       optional pre-allocated output tensor
    training:  bool, True -> apply dropout, False -> skip dropout
    """
    # 1) Z1 = input @ weight1
    z1 = triton_matmul(input, weight1)
    # 2) Z2 = softmax(Z1)
    z2 = triton_softmax(z1)
    # 3) Z3 = dropout(Z2, dropout_p)
    z3 = dropout(z2, p=dropout_p, training=training)
    # 4) Z4 = Z3 @ weight2
    z4 = triton_matmul(z3, weight2)
    # 5) Y = LayerNorm(Z4 + residual)
    z4_res = z4 + residual
    y = triton_layer_norm(z4_res, eps=eps)

    if out is not None:
        out.copy_(y)
        return out
    return y
