import torch
import triton
import triton.language as tl

@triton.jit
def _tanh_linear_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_outm, stride_outn,
    stride_b,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr
):
    """Compute tanh((X @ W^T) + b) for a single tile."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute tile's starting indices for M and N dimensions
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Create a 2D range for the block
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)

    # K-loop range
    offs_k = tl.arange(0, BLOCK_K)
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Loop over K dimension
    # Each iteration loads a sub-block of A and B, multiplies and accumulates
    for k_start in range(0, K, BLOCK_K):
        # Actual size of the K-block (may be smaller on the last iteration)
        k_size = tl.max(tl.min(K - k_start, BLOCK_K), 0)

        # Load X
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm) + ((k_start + offs_k[None, :]) * stride_xk)
        # Load W
        w_ptrs = W_ptr + ((n_start + tl.arange(0, BLOCK_N))[None, :] * stride_wn) \
                         + ((k_start + offs_k[:, None]) * stride_wk)

        # Mask to avoid out-of-bounds
        x_mask = (offs_m[:, None] < M) & ((k_start + offs_k[None, :]) < K)
        w_mask = ((n_start + tl.arange(0, BLOCK_N))[None, :] < N) & ((k_start + offs_k[:, None]) < K)

        x_block = tl.where(x_mask, tl.load(x_ptrs, mask=x_mask, other=0.), 0.)
        w_block = tl.where(w_mask, tl.load(w_ptrs, mask=w_mask, other=0.), 0.)

        # Compute partial matmul for the k_size slice
        acc += tl.dot(x_block, w_block)

    # Add bias if present
    if HAS_BIAS:
        bias_ptrs = B_ptr + (offs_n[None, :]) * stride_b
        bias_mask = (offs_n[None, :] < N)
        bias_vals = tl.where(bias_mask, tl.load(bias_ptrs, mask=bias_mask, other=0.), 0.)
        acc += bias_vals

    # Apply Tanh activation
    acc = tl.tanh(acc)

    # Write back
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_outm) + (offs_n[None, :] * stride_outn)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


def tanh_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """
    Applies a linear transformation to the input tensor followed by a Tanh activation.

    Args:
        input (Tensor): The input tensor of shape (*, in_features), where * represents
                        any number of additional dimensions.
        weight (Tensor): The weight matrix of shape (out_features, in_features).
        bias (Tensor, optional): The bias tensor of shape (out_features). Default: None

    Returns:
        Tensor: Output tensor after performing tanh(linear(input, weight, bias)).
    """
    # Flatten input to 2D (M x K), keep track of original shape except last dimension
    orig_shape = input.shape
    in_features = weight.shape[1]
    out_features = weight.shape[0]

    # Check shapes
    assert orig_shape[-1] == in_features, \
        f"Expected input's last dimension ({orig_shape[-1]}) to match in_features ({in_features})."
    if bias is not None:
        assert bias.numel() == out_features, \
            f"Expected bias size ({bias.numel()}) to match out_features ({out_features})."

    M = 1
    for s in orig_shape[:-1]:
        M *= s
    K = in_features
    N = out_features

    x_2d = input.view(M, K)
    output = torch.empty((M, N), dtype=input.dtype, device=input.device)

    # Launch config
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N)
    )

    # Strides
    stride_xm = x_2d.stride(0)
    stride_xk = x_2d.stride(1)
    stride_wn = weight.stride(0)
    stride_wk = weight.stride(1)
    stride_outm = output.stride(0)
    stride_outn = output.stride(1)
    stride_b = bias.stride(0) if bias is not None else 0

    _tanh_linear_kernel[grid](
        x_2d, weight, bias if bias is not None else x_2d, output,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_outm, stride_outn,
        stride_b,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        HAS_BIAS=(bias is not None)
    )

    # Reshape back to original shape, replacing last dimension with out_features
    out_shape = list(orig_shape[:-1]) + [N]
    return output.view(*out_shape)
