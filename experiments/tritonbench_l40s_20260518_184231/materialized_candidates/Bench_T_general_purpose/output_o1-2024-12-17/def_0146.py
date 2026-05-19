import triton
import triton.language as tl

@triton.jit
def _elu_linear_kernel(
    INPUT_PTR,       # ptr float32, shape [batch, in_features]
    WEIGHT_PTR,      # ptr float32, shape [in_features, out_features]
    BIAS_PTR,        # ptr float32, shape [out_features] or None
    OUTPUT_PTR,      # ptr float32, shape [batch, out_features]
    batch_size,      # int32
    in_features,     # int32
    out_features,    # int32
    alpha,           # float32
    apply_bias,      # bool
    BLOCK_M: tl.constexpr,  # block size for M dimension
    BLOCK_N: tl.constexpr,  # block size for N dimension
    num_warps: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Starting indices for the sub-block
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Create a 2D range for the sub-block
    rm = m_start + tl.arange(0, BLOCK_M)
    rn = n_start + tl.arange(0, BLOCK_N)

    # Pointers for loading/storing
    input_offset = rm[:, None] * in_features + tl.arange(0, BLOCK_M)[None, :]
    weight_offset = tl.arange(0, BLOCK_N)[:, None] + rn[None, :] * in_features

    # Initialize accumulator
    accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension (in_features) in chunks of BLOCK_K
    BLOCK_K = 32
    for k_start in range(0, in_features, BLOCK_K):
        k_range = tl.arange(0, BLOCK_K)
        k_idx = k_start + k_range
        # Mask for valid K
        k_mask = k_idx < in_features

        # Load a block of input and weight
        input_ptrs = INPUT_PTR + (rm[:, None] * in_features + k_idx[None, :])
        weight_ptrs = WEIGHT_PTR + (k_idx[:, None] * out_features + rn[None, :])

        input_block = tl.load(input_ptrs, mask=k_mask[None, :], other=0.0)
        weight_block = tl.load(weight_ptrs, mask=k_mask[:, None], other=0.0)

        # Accumulate partial
        accum += tl.dot(input_block, weight_block)

    # Optionally add bias
    if apply_bias:
        bias_vals = tl.load(BIAS_PTR + rn, mask=rn < out_features, other=0.0)
        accum += bias_vals[None, :]

    # Apply ELU
    # out = x if x>0 else alpha*(exp(x) - 1)
    mask_pos = accum > 0
    elu_pos = accum
    elu_neg = alpha * (tl.exp(accum) - 1)
    accum = tl.where(mask_pos, elu_pos, elu_neg)

    # Store output
    out_ptrs = OUTPUT_PTR + rm[:, None] * out_features + rn[None, :]
    # Mask for valid range
    out_mask_m = rm < batch_size
    out_mask_n = rn < out_features
    tl.store(out_ptrs, accum, mask=out_mask_m[:, None] & out_mask_n[None, :])


def elu_linear(input, weight, bias=None, alpha=1.0, inplace=False):
    """
    elu_linear(input, weight, bias=None, alpha=1.0, inplace=False) -> Tensor
    input (Tensor): The input tensor for the linear layer.
    weight (Tensor): The weight tensor for the linear transformation.
    bias (Tensor, optional): The bias tensor for the linear transformation. Default: None.
    alpha (float, optional): The α parameter for the ELU function. Default: 1.0.
    inplace (bool, optional): Whether to apply ELU in-place. Default: False.
    """
    import torch

    # Shapes
    batch_size, in_features = input.shape
    w_in, out_features = weight.shape

    assert in_features == w_in, "Input's in_features must match weight's in_features."
    if bias is not None:
        assert bias.shape[0] == out_features, "Bias shape must match out_features."

    # Prepare output
    if inplace:
        output = input
    else:
        output = torch.empty((batch_size, out_features), device=input.device, dtype=input.dtype)

    # Grid
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (
        (batch_size + BLOCK_M - 1) // BLOCK_M,
        (out_features + BLOCK_N - 1) // BLOCK_N
    )

    triton.run(
        _elu_linear_kernel,
        args=[
            input.data_ptr(),
            weight.data_ptr(),
            bias.data_ptr() if bias is not None else 0,
            output.data_ptr(),
            batch_size,
            in_features,
            out_features,
            alpha,
            bool(bias is not None)
        ],
        grid=grid,
        num_warps=4,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return output
