input_row_stride,
    grad_input_ptr: tl.pointer_type,
    grad_weight_accum_ptr: tl.pointer_type,
    num_elements,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    """
    Compute the RMSNorm gradients.
    This kernel is based on the 1-pass RMSNorm algorithm described in the original paper
    https://arxiv.org/pdf/1910.07467.pdf by Ilya Loshchilov and Frank Hutter.
    """
    row_idx = tl.program_id(axis=0)
    col_offsets = tl.arange(0, block_size)
    row_mask = col_offsets < num_elements

    input_ptr += row_idx * input_row_stride
    grad_output_ptr += row_idx * input_row_stride
    grad_input_ptr += row_idx * input_row_stride
    weight_ptr += col_offsets
    grad_weight_accum_ptr += col_offsets

    input_row = tl.load(input_ptr + col_offsets, mask=row_mask, other=0.0).to(tl.float32)
    grad_output_row = tl.load(grad_output_ptr + col_offsets, mask=row_mask, other=0.0).to(tl.float32)
    weight_row = tl.load(weight_ptr, mask=row_mask).to(tl.float32)

    input_row_dtype = input_row.dtype
    mean_square = tl.sum(input_row * input_row) / num_elements
    rstd = 1.0 / tl.sqrt(mean_square + eps)

    # Compute the gradient of the inputs
    grad_weight_accum = tl.sum(input_row * grad_output_row * rstd * input_row, axis=0)
    tl.store(grad_weight_accum_ptr, grad_weight_accum)
    grad_input_row = (
        grad_output_row * rstd * weight_row * (1.0 - 0.5 * input_row * input_row * (rstd * rstd * rstd))
    )
    tl.store(grad_input_ptr + col_offsets, grad_input_row, mask=row_mask)

def rmsnorm_forward(x, weight, eps):
    assert weight.dtype == x.dtype, "It must be same type between weight and x"
    assert x.strides[-1] == 1, "The last dimension of x must be contiguous"
    assert x.shape[-1] < 65536, "The feature dimension should be smaller than 64KB"
    y = torch.empty_like(x)
    N = x.shape[-1]
    M = x.numel() // N
    L = math.ceil(math.sqrt(N))
    rstd = torch.empty((M, L), dtype=x.dtype, device=x.device)

    # Create a grid with two dimensions: the first for rows, the second for batch
    grid = lambda META: (M, 1)
    rmsnorm_fwd_kernel[grid](
        x,
        y,
        weight,
        rstd,
        x.stride(0),
        x.stride(1),
        L,
        N,
        eps,
        BLOCK_SIZE=triton.next_power_of_2(N),
    )
    return y, rstd

def rmsnorm_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    y_grad: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Compute RMSNorm backward
    """
    assert weight.dtype == x.dtype, "It must be same type between weight and x"
    assert x.strides[-1] == 1, "The last dimension of x must be contiguous"
    assert x.shape[-1] < 65536, "The feature dimension should be smaller than 64KB"
    x_grad = torch.empty_like(x)

    assert y_grad.stride(-1) == 1, "The last dimension of y_grad must be contiguous"
    assert y_grad.shape[-1] == x.shape[-1], "The feature dimension must be the same"

    L = math.ceil(math.sqrt(x.shape[-1]))
    num_blocks = x.numel() // x.shape[-1]

    block_size = triton.next_power_of_2(x.shape[-1])
    grid = (num_blocks, 1)

    rmsnorm_bwd_kernel[grid](
        x,
        weight,
        y_grad,
        x.stride(0),
        x_grad,
        torch.empty((1,), dtype=torch.float32, device=x.device),
        x.numel(),
        eps,
        block_size,
    )
    return x_grad
