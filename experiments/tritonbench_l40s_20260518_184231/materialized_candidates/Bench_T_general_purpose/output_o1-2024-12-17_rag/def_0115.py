import torch
import triton
import triton.language as tl

@triton.jit
def _tanh(x):
    epos = tl.exp(x)
    eneg = tl.exp(-x)
    return (epos - eneg) / (epos + eneg)

@triton.jit
def _kernel_tanh_linear_fwd(
    OUT,          # out_ptr
    X,            # x_ptr
    W,            # w_ptr
    B,            # b_ptr
    M,            # number of rows of X (and OUT)
    N,            # number of columns of W (and OUT)
    K,            # shared dimension for matmul
    stride_out_m, # OUT.stride(0)
    stride_xm,    # X.stride(0)
    stride_xk,    # X.stride(1)
    stride_wk,    # W.stride(1)
    stride_wn,    # W.stride(0)
    BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute out = tanh( X @ W^T + bias ).
    - X is [M, K]
    - W is [N, K] (we treat it as W^T for the dot so each row is one of K, each col is one of N)
    - B is optional [N]
    - OUT is [M, N]
    Shapes must match so that X[K] = W[K].
    """
    pid = tl.program_id(axis=0)
    # number of blocks along the M dimension
    num_m_blocks = (M + BLOCK_M - 1) // BLOCK_M
    # number of blocks along the N dimension
    num_n_blocks = (N + BLOCK_N - 1) // BLOCK_N
    # each program handles one (block_m, block_n) tile
    block_m = pid // num_n_blocks
    block_n = pid % num_n_blocks

    # compute the row/col range for the block
    rm = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = block_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # pointers to X block and W block
    x_ptrs = X + (rm[:, None] * stride_xm + tl.arange(0, BLOCK_K)[None, :] * stride_xk)
    w_ptrs = W + (rn[None, :] * stride_wn + tl.arange(0, BLOCK_K)[:, None] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # k_loop: loop over K dimension in chunks of BLOCK_K
    # we already allocated a range(0, BLOCK_K), so we do it repeatedly if needed
    for k_offset in range(0, K, BLOCK_K):
        # effective K size for this iteration
        k_eff = tl.arange(0, BLOCK_K) + k_offset
        # mask out-of-bounds loads
        k_mask = k_eff < K

        # load from X and W
        x = tl.load(x_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)
        w = tl.load(w_ptrs, mask=k_mask[:, None], other=0.0).to(tl.float32)

        acc += tl.dot(x, w)

        # advance pointers
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # add bias if provided
    if BIAS:
        bias_vals = tl.load(B + rn, mask=rn < N, other=0.0).to(tl.float32)
        acc += bias_vals[None, :]

    # apply tanh element-wise
    acc = _tanh(acc)

    # store to output
    out_ptrs = OUT + (rm[:, None] * stride_out_m + rn[None, :])
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(out_ptrs, acc.to(OUT.dtype.element_ty), mask=mask)

def tanh_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """
    Performs a linear transformation of the form y = x * weight^T + bias
    followed by an element-wise Tanh activation:
        out = tanh( y ) = tanh( xA^T + b ).
    
    Args:
        input (Tensor): shape (*, in_features)
        weight (Tensor): shape (out_features, in_features)
        bias (Tensor, optional): shape (out_features), default None
    
    Returns:
        Tensor of shape (*, out_features)
    """
    # check shapes and dtypes
    assert input.dim() >= 2, "Input must have at least 2 dimensions"
    assert weight.dim() == 2, "Weight must be 2D"
    if bias is not None:
        assert bias.dim() == 1, "Bias must be 1D"
        assert bias.shape[0] == weight.shape[0], "Bias size must match out_features"
    batch_shape = input.shape[:-1]
    in_features = input.shape[-1]
    out_features = weight.shape[0]
    assert in_features == weight.shape[1], "Input size and weight size mismatch"

    # flatten the leading dimensions
    M = int(torch.prod(torch.tensor(batch_shape)))  # total batch
    x_reshaped = input.reshape(M, in_features)

    # create output
    output = torch.empty((M, out_features), device=input.device, dtype=input.dtype)
    # handle strides
    # ensure contiguity if needed
    if x_reshaped.stride(0) != in_features:
        x_reshaped = x_reshaped.contiguous()
    if weight.stride(1) != 1:
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    # define block sizes
    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_K = 32

    # grid: how many program ids we need
    grid = ((M + BLOCK_M - 1) // BLOCK_M) * ((out_features + BLOCK_N - 1) // BLOCK_N)

    _kernel_tanh_linear_fwd[grid](
        output,
        x_reshaped,
        weight,
        bias if bias is not None else x_reshaped,  # dummy pointer if bias is None
        M,
        out_features,
        in_features,
        output.stride(0),
        x_reshaped.stride(0),
        x_reshaped.stride(1),
        weight.stride(1),
        weight.stride(0),
        BIAS=(bias is not None),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # reshape back
    return output.reshape(*batch_shape, out_features)
