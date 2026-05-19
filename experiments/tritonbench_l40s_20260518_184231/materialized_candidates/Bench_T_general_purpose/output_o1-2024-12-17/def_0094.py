import triton
import triton.language as tl
import torch


@triton.jit
def _matmul_bias_kernel(
    A_ptr, B_ptr, C_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr
):
    """
    Compute C = A x B + bias (if HAS_BIAS is True)

    A is of shape (M, K)
    B is of shape (K, N)
    bias is of shape (N,) or None
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create block offset
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm_mask = rm < M
    rn_mask = rn < N

    # Pointer arithmetic for the output
    offsC = rm[:, None] * stride_cm + rn[None, :] * stride_cn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k_offs in range(0, K, BLOCK_K):
        kk = tl.arange(0, BLOCK_K)
        k_mask = k_offs + kk < K

        # Offsets for A and B
        a_offs = rm[:, None] * stride_am + (k_offs + kk)[None, :] * stride_ak
        b_offs = (k_offs + kk)[:, None] * stride_bk + rn[None, :] * stride_bn

        # Load A and B from DRAM
        a = tl.where(rm_mask[:, None] & k_mask[None, :],
                     tl.load(A_ptr + a_offs, mask=rm_mask[:, None] & k_mask[None, :], other=0.0),
                     0.0)
        b = tl.where(k_mask[:, None] & rn_mask[None, :],
                     tl.load(B_ptr + b_offs, mask=k_mask[:, None] & rn_mask[None, :], other=0.0),
                     0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Optionally add bias
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + rn, mask=rn_mask, other=0.0)
        acc = tl.where(rm_mask[:, None] & rn_mask[None, :],
                       acc + bias_vals[None, :],
                       acc)

    # Store the result
    c = tl.where(rm_mask[:, None] & rn_mask[None, :], acc, 0.0)
    tl.store(C_ptr + offsC, c, mask=rm_mask[:, None] & rn_mask[None, :])


@triton.jit
def _sigmoid_dropout_kernel(
    ptr_in_out, ptr_mask,
    n_elements,
    prob,
    training: tl.constexpr,
    inplace: tl.constexpr
):
    """
    Apply sigmoid activation, then dropout if training=True
    """
    pid = tl.program_id(0)
    block_size = 1024
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements

    x = tl.load(ptr_in_out + offsets, mask=mask, other=0.0)

    # Sigmoid
    x = 1.0 / (1.0 + tl.exp(-x))

    if training:
        # Generate dropout mask
        rand = tl.load(ptr_mask + offsets, mask=mask, other=1.0)
        drop_mask = rand >= prob
        # Apply dropout
        x = x * drop_mask * (1.0 / (1.0 - prob))

    tl.store(ptr_in_out + offsets, x, mask=mask)


def dropout_sigmoid_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    p=0.5,
    training=True,
    inplace=False
) -> torch.Tensor:
    """
    Applies (input * weight^T + bias) -> sigmoid -> optional dropout.
    """
    # Ensure input is 2D or flatten the leading dimensions
    if input.dim() > 2:
        input_reshape = input.view(-1, input.size(-1))
    else:
        input_reshape = input

    M, K = input_reshape.shape
    N = weight.shape[0]

    # Prepare output tensor
    out_shape = (M, N)
    if inplace and (M, K) == (M, N):
        # Only allow in-place if shape is unchanged, but shape changes from NxK to NxN
        # for a typical linear transform - if they differ, we allocate new
        output = input_reshape
    else:
        output = torch.empty(out_shape, dtype=input_reshape.dtype, device=input_reshape.device)

    # Launch Triton kernel for matmul + bias
    grid = lambda META: ( (M + META['BLOCK_M'] - 1) // META['BLOCK_M'],
                          (N + META['BLOCK_N'] - 1) // META['BLOCK_N'] )

    HAS_BIAS = bias is not None

    triton.run(
        _matmul_bias_kernel,
        grid=grid,
        args=[
            input_reshape, weight, output, bias if HAS_BIAS else torch.empty(0, device=input.device),
            M, N, K,
            input_reshape.stride(0), input_reshape.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
            64, 64, 32,  # Tunable BLOCK sizes
            HAS_BIAS
        ],
        num_warps=4,
        num_stages=2
    )

    # Now apply sigmoid and dropout if training
    total_elems = output.numel()
    if total_elems > 0:
        block_size = 1024
        grid2 = ( (total_elems + block_size - 1) // block_size, )

        # Dropout requires a random mask
        # We'll create a float mask in [0, 1)
        if training:
            mask = torch.rand_like(output)
        else:
            mask = torch.ones_like(output)

        triton.run(
            _sigmoid_dropout_kernel,
            grid=grid2,
            args=[
                output, mask,
                total_elems,
                p,
                training,
                inplace
            ],
            num_warps=4
        )

    # Reshape back if input was reshaped
    if input.dim() > 2:
        out = output.view(*list(input.shape[:-1]), N)
    else:
        out = output

    return out
