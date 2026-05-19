import triton
import triton.language as tl

@triton.jit
def _softplus_linear_kernel(
    INPUT_PTR, WEIGHT_PTR, BIAS_PTR, OUTPUT_PTR,
    N, M, K,
    stride_inN, stride_inK,
    stride_wM, stride_wK,
    stride_outN, stride_outM,
    stride_b, has_bias,
    beta, threshold,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Computes out = Softplus( (input @ weight^T) + bias ), applying a block-level matmul, then a softplus.
    Dimensions: 
      input:  [N, K]
      weight: [M, K]
      bias:   [M] or None
      output: [N, M]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create ranges for m, n
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create a pointer mask for valid m, n
    mask_m = rm < M
    mask_n = rn < N

    # Broadcast to 2D
    RM = tl.broadcast_to(rm[:, None], [BLOCK_M, BLOCK_N])
    RN = tl.broadcast_to(rn[None, :], [BLOCK_M, BLOCK_N])

    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # k loop
    # Each program will iterate over K in steps of BLOCK_K
    rk_loop = range(0, K, BLOCK_K)
    for rk_start in rk_loop:
        # Offsets for input and weight
        rk_offsets = tl.arange(0, BLOCK_K)
        k_mask = rk_start + rk_offsets < K

        # Load input: shape [BLOCK_N, BLOCK_K] transposed to [BLOCK_K, BLOCK_N] for broadcasting
        in_ptrs = INPUT_PTR + (RN[None, :] * stride_inN) + ((rk_start + rk_offsets)[:, None] * stride_inK)
        inp = tl.load(in_ptrs, mask=(mask_n[None, :] & k_mask[:, None]), other=0.0)

        # Load weight: shape [BLOCK_M, BLOCK_K]
        w_ptrs = WEIGHT_PTR + (RM[:, None] * stride_wM) + ((rk_start + rk_offsets)[None, :] * stride_wK)
        wgt = tl.load(w_ptrs, mask=(mask_m[:, None] & k_mask[None, :]), other=0.0)

        # Partial matmul
        acc += tl.dot(wgt, inp)

    # Add bias if present
    if has_bias:
        # broadcast bias values to [BLOCK_M, BLOCK_N]
        bias_vals = tl.load(BIAS_PTR + rm * stride_b, mask=mask_m, other=0.0)
        acc += bias_vals[:, None]

    # Apply softplus
    # softplus(x) = (1/beta) * log(1 + exp(beta * x)) for x <= threshold, else x
    # note: threshold is scaled by 'beta' inside the condition
    acc_scaled = beta * acc
    out = tl.where(acc_scaled > beta * threshold,
                   acc, 
                   (1.0 / beta) * tl.log(1.0 + tl.exp(acc_scaled)))

    # Store results
    out_ptrs = OUTPUT_PTR + (RN[None, :] * stride_outN) + (RM[:, None] * stride_outM)
    tl.store(out_ptrs, out, mask=(mask_m[:, None] & mask_n[None, :]))


def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
    """
    softplus_linear(input, weight, bias=None, beta=1, threshold=20) -> Tensor
    Applies a linear transformation to 'input' using 'weight' and (optionally) 'bias',
    followed by Softplus element-wise with parameters 'beta' and 'threshold'.
    """
    import torch
    # Shapes
    assert input.ndim == 2, "Input must be 2D"
    assert weight.ndim == 2, "Weight must be 2D"
    N, K = input.shape
    Mw, Kw = weight.shape
    # We expect weight in [M, K] for (input @ weight^T). So K == Kw, M = Mw
    assert K == Kw, "Inner dimensions must match"
    M = Mw

    has_bias = 1 if bias is not None else 0
    if bias is not None:
        assert bias.shape[0] == M, "Bias shape must match out_features"

    # Allocate output
    out = torch.empty((N, M), dtype=input.dtype, device=input.device)

    # Strides
    stride_inN = input.stride(0)
    stride_inK = input.stride(1)

    stride_wM = weight.stride(0)
    stride_wK = weight.stride(1)

    stride_outN = out.stride(0)
    stride_outM = out.stride(1)

    # Bias
    if bias is not None:
        stride_b = bias.stride(0)
    else:
        stride_b = 0

    # Grid
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        ( (M + BLOCK_M - 1) // BLOCK_M ),
        ( (N + BLOCK_N - 1) // BLOCK_N )
    )

    _softplus_linear_kernel[grid](
        input, weight, bias if bias is not None else input, out,
        N, M, K,
        stride_inN, stride_inK,
        stride_wM, stride_wK,
        stride_outN, stride_outM,
        stride_b, has_bias,
        beta, threshold,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return out
