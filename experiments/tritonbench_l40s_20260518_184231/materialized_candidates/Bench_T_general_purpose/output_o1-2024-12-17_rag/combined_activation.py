import triton
import triton.language as tl
import torch

@triton.jit
def _combined_activation_kernel(
    # Pointers
    X_PTR,        # (*, N, D_in) flattened into (M, K)
    W1_PTR,       # (K, D_out)
    W2_PTR,       # broadcastable to (M, D_out)
    B_PTR,        # broadcastable to (M, D_out)
    OUT_PTR,      # (M, D_out)
    # Dimensions
    M,            # total rows after flattening (*, N)
    K,            # D_in
    D_OUT,        # D_out
    # Strides for X
    STRIDE_XM,    # how many elements to move in X when we go down 1 row
    STRIDE_XK,    # how many elements to move in X when we move in the K dimension
    # Strides for W1
    STRIDE_W1K,   # how many elements to move in W1 when we move down 1 row in K
    STRIDE_W1N,   # how many elements to move in W1 when we move along D_out
    # Strides for W2, B (they may be 0 for broadcast along some dim)
    STRIDE_W2M,
    STRIDE_W2N,
    STRIDE_BM,
    STRIDE_BN,
    # Strides for OUT
    STRIDE_OM,
    STRIDE_ON,
    # Meta-parameters
    BLOCK_M: tl.constexpr,  # block size in the M dimension
    BLOCK_N: tl.constexpr,  # block size in the D_out dimension
    BLOCK_K: tl.constexpr   # block size in the K dimension
):
    """
    Computes:
      OUT = (tanh(sigmoid(X @ W1)) * W2) + B
    Where:
      X has shape (M, K), W1 has shape (K, D_OUT),
      W2 and B are broadcastable to (M, D_OUT),
      OUT has shape (M, D_OUT).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets in M and N for this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create pointers for X and W1 sub-block accumulations
    # Each thread block will load a tile of X of shape [BLOCK_M, BLOCK_K]
    # and a tile of W1 of shape [BLOCK_K, BLOCK_N]
    # Then do a matmul accumulation.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # k-range to iterate
    for k_start in range(0, K, BLOCK_K):
        range_k = k_start + tl.arange(0, BLOCK_K)
        # Load X tile
        x_ptrs = X_PTR + (offs_m[:, None] * STRIDE_XM) + (range_k[None, :] * STRIDE_XK)
        x_mask = (offs_m < M) & (range_k < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

        # Load W1 tile
        w1_ptrs = W1_PTR + (range_k[:, None] * STRIDE_W1K) + (offs_n[None, :] * STRIDE_W1N)
        w1_mask = (range_k < K) & (offs_n < D_OUT)
        w1_tile = tl.load(w1_ptrs, mask=w1_mask, other=0.0).to(tl.float32)

        # Matmul accumulate for this K-chunk
        acc += tl.dot(x_tile, w1_tile)

    # Now apply sigmoid, tanh, multiply by W2, then add B
    # Indices for the final [M, D_out] block
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < D_OUT)

    # Sigmoid
    # sig(z) = 1 / (1 + exp(-z))
    acc = 1.0 / (1.0 + tl.exp(-acc))

    # Tanh
    # tanh(z) = (e^z - e^-z) / (e^z + e^-z)
    acc = tl.tanh(acc)

    # Load W2
    # W2 is broadcastable; use stride 0 where needed
    w2_ptrs = W2_PTR + (offs_m[:, None]*STRIDE_W2M) + (offs_n[None, :]*STRIDE_W2N)
    w2_load = tl.load(w2_ptrs, mask=out_mask, other=1.0).to(tl.float32)
    acc = acc * w2_load

    # Load bias
    b_ptrs = B_PTR + (offs_m[:, None]*STRIDE_BM) + (offs_n[None, :]*STRIDE_BN)
    b_load = tl.load(b_ptrs, mask=out_mask, other=0.0).to(tl.float32)
    acc = acc + b_load

    # Store to OUT
    out_ptrs = OUT_PTR + (offs_m[:, None] * STRIDE_OM) + (offs_n[None, :] * STRIDE_ON)
    tl.store(out_ptrs, acc, mask=out_mask)

def combined_activation(input, weight1, weight2, bias, *, out=None):
    """
    combined_activation(input, weight1, weight2, bias, *, out=None) -> Tensor
    
    Performs Y = (tanh(sigmoid(X @ W1)) * W2) + B

    Arguments:
        input (Tensor):  shape (*, N, D_in)
        weight1 (Tensor): shape (D_in, D_out)
        weight2 (Tensor): broadcastable to intermediate shape (*, N, D_out)
        bias (Tensor): broadcastable to shape (*, N, D_out)
        out (Tensor, optional): Output tensor. Ignored if None.
    
    Returns:
        Tensor of shape (*, N, D_out).
    """
    # Flatten any leading batch dimensions plus N into a single M dimension
    # so that input becomes [M, D_in].
    in_shape = input.shape
    *batch_dims, N, D_in = in_shape
    D_out = weight1.shape[1]

    # Verify matrix multiply consistency
    assert D_in == weight1.shape[0], "Incompatible dims for matmul"

    M = 1
    for d in batch_dims:
        M *= d
    M *= N

    X_2d = input.reshape(M, D_in)
    # Prepare output
    if out is None:
        out = input.new_empty(M, D_out)
    else:
        out_shape = list(batch_dims) + [N, D_out]
        assert out.shape == tuple(out_shape), "out tensor has incorrect shape"

    # Enforce contiguous for safety
    X_2d = X_2d.contiguous()
    W1_c = weight1.contiguous()
    W2_c = weight2.contiguous()
    B_c  = bias.contiguous()
    out_c = out.reshape(M, D_out).contiguous()

    # Strides
    stride_xm = X_2d.stride(0)
    stride_xk = X_2d.stride(1)
    stride_w1k = W1_c.stride(0)
    stride_w1n = W1_c.stride(1)

    # For broadcasting W2 and B
    # If shape is [M, D_out], stride_w2m = w2.stride(0), etc.
    # If broadcast along M, stride may be 0 on that dimension; same for D_out.
    # We'll compute strides carefully against the final shape [M, D_out].
    # Expand weight2 to (M, D_out), then read out strides
    # Expand bias similarly.
    w2_expanded = W2_c
    if list(w2_expanded.shape) != [M, D_out]:
        w2_expanded = w2_expanded.broadcast_to([M, D_out])
    b_expanded = B_c
    if list(b_expanded.shape) != [M, D_out]:
        b_expanded = b_expanded.broadcast_to([M, D_out])

    # Contiguous expansions
    w2_expanded = w2_expanded.contiguous()
    b_expanded  = b_expanded.contiguous()
    
    stride_w2m, stride_w2n = w2_expanded.stride()
    stride_bm,  stride_bn  = b_expanded.stride()

    stride_om, stride_on = out_c.stride()

    # Define block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    # 2D grid
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(D_out, META['BLOCK_N'])
    )
    
    _combined_activation_kernel[grid](
        X_2d, W1_c, w2_expanded, b_expanded, out_c,
        M, D_in, D_out,
        stride_xm, stride_xk,
        stride_w1k, stride_w1n,
        stride_w2m, stride_w2n,
        stride_bm,  stride
