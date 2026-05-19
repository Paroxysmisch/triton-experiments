import torch
import triton
import triton.language as tl

@triton.jit
def kernel_tanh_linear(
    C,  # Pointers to matrices
    A,
    B,
    bias,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_cm,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    BIAS: tl.constexpr,
):
    """
    Kernel for computing Out = tanh(A x W + C)
    - Input has shape (M, K)
    - Weight has shape (K, N)
    - Bias has shape (N,)
    - Output has shape (M, N)
    This kernel will consolidate over K
    """
    pid = tl.program_id(axis=0)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    # re-order program ID for better L2 performance
    width = grid_m * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * grid_m, grid_m)
    pid_m = group_id * grid_m + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices
    # for rows (resp. col) of C
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # trick to avoid masking on M and N axis
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b)

        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # Adding bias if provided
    if BIAS:
        bias = tl.load(bias + rn, mask=rn < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    # Apply Tanh activation
    acc = tl.tanh(acc)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # write back result
    C = C + rm[:, None] * stride_cm + rn[None, :]
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, acc, mask=mask)

def tanh_linear(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Applies a linear transformation to the input tensor followed by a Tanh activation function.
    
    :param input: The input tensor of shape `(*, in_features)`, where `*` represents any number of additional dimensions.
    :param weight: The weight matrix of shape `(out_features, in_features)`.
    :param bias: The optional bias tensor of shape `(out_features)`. Default: None.
    :return: The result tensor of shape `(*, out_features)`.
    """
    batch_shape, in_features = input.shape[:-1], input.shape[-1]
    batch_dim = batch_shape.numel()
    input_reshaped = input.reshape(batch_dim, in_features)

    if input_reshaped.stride(0) > 1 and input_reshaped.stride(1) > 1:
        input_reshaped = input_reshaped.contiguous()
    if weight.stride(0) > 1 and weight.stride(1) > 1:
        weight = weight.contiguous()
    bias = bias.contiguous() if bias is not None else None

    assert (
        input.dtype == weight.dtype
    ), f"Input and weight must have the same dtype, got {input.dtype} and {weight.dtype}"
    if bias is not None:
        assert (
            input.dtype == bias.dtype
        ), f"Input and bias must have the same dtype, got {input.dtype} and {bias.dtype}"
    assert (
        input_reshaped.shape[1] == weight.shape[1]
    ), f"Incompatible dimensions: {input_reshaped.shape} - {weight.shape}"

    assert (
        bias is None or bias.shape[0] == weight.shape[0]
    ), "Incompatible dimensions in between weight and bias"

    M, K = input_reshaped.shape
    N, K = weight.shape

    output = torch.empty((M, N), device=input.device, dtype=input.dtype)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)  # noqa

    kernel_tanh_linear[grid](
        output,
        input_reshaped,
        weight,  # data ptrs
        bias if bias is not None else input,  # auto skip bias if not present
        M,  # shapes
        N,
        K,
        stride_cm=output.stride(0),  # strides
        stride_am=input_reshaped.stride(0),
        stride_ak=input_reshaped.stride(1),
        stride_bk=weight.stride(1),
        stride_bn=weight.stride(0),
        BIAS=bias is not None,  # optional fused bias
        BLOCK_M=128,  # block size for M
        BLOCK_N=128,  # block size for N
        BLOCK_K=32,  # block size for K
        SPLIT_K=1,  # split K not used, not performant with activation, kept because early_config_prune is expecting it
        EVEN_K=True,  # assume K is a multiple of BLOCK_K
    )

    return output.reshape(*batch_shape, output.shape[-1])
