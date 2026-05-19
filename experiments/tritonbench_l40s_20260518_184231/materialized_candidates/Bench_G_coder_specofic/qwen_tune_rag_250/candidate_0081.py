+ batch * L

    # Compute mean and variance
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + base_idx + cols, mask=cols < N, other=0.0).to(tl.float32)
    x = tl.where(cols < N, x, 0.0)
    mean = tl.sum(x, axis=0) / N
    xbar = tl.where(cols < N, x - mean, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Store rstd for later use in backward pass
    tl.store(Rstd + row, rstd)

    # Normalize and apply rmsnorm
    mask = cols < N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    y = x * rstd * w
    # Write-back to Y
    tl.store(Y + base_idx + cols, y, mask=mask)


@triton.jit
def rmsnorm_bwd_kernel(
    input_ptr,
    weight_ptr,
    grad_output_ptr,
    input_row_stride,
    grad_input_ptr,
    grad_weight_accum_ptr,
    num_elements,
    eps,
    block_size: tl.constexpr,
):
    """
    Implements a backward kernel for root mean square layer normalization.
    
    Parameters:
    input_ptr (tl.tensor): Pointer to input tensor.
    weight_ptr (tl.tensor): Pointer to weights tensor.
    grad_output_ptr (tl.tensor): Pointer to gradient output tensor.
    input_row_stride (int): Stride for rows in the input tensor.
    grad_input_ptr (tl.tensor): Pointer to gradient input tensor.
    grad_weight_accum_ptr (tl.tensor): Pointer to accumulated gradients for weights.
    num_elements (int): Total number of elements in the tensor.
    eps (float): Small epsilon value for numerical stability.
    block_size (tl.constexpr): Block size for computation.
    """
    # Initialize offsets for the current program
    row_start = tl.program_id(0) * block_size
    grad_input_offset = row_start * input_row_stride
    input_offset = row_start * input_row_stride
    grad_output_offset = row_start * input_row_stride
    weight_offset = tl.program_id(0) * block_size
    grad_weight_accum_offset = tl.program_id(0) * block_size

    # Load input, weights, and gradients
    input_row = tl.load(
        input_ptr + input_offset,
        mask=(input_offset + tl.arange(0, block_size)) < num_elements,
        other=0.0,
    ).to(tl.float32)
    grad_output_row = tl.load(
        grad_output_ptr + grad_output_offset,
        mask=(grad_output_offset + tl.arange(0, block_size)) < num_elements,
        other=0.0,
    ).to(tl.float32)
    weight_row = tl.load(
        weight_ptr + weight_offset,
        mask=(weight_offset + tl.arange(0, block_size)) < block_size,
        other=0.0,
    ).to(tl.float32)

    # Compute intermediate values for backward pass
    rstd = 1.0 / tl.sqrt(
        tl.sum(input_row * input_row) / block_size + eps
    )
    grad_weight_row = grad_output_row * (input_row * rstd)
    grad_input_row = grad_output_row * rstd * weight_row

    # Accumulate grad_weight
    tl.atomic_add(
        grad_weight_accum_ptr + grad_weight_accum_offset,
        grad_weight_row,
        mask=(grad_weight_accum_offset + tl.arange(0, block_size)) < num_elements,
    )

    # Store the result in grad_input_ptr
    tl.store(
        grad_input_ptr + grad_input_offset,
        grad_input_row,
        mask=(grad_input_offset + tl.arange(0, block_size)) < num_elements,
    )


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        eps=1e-6,
        track_running_stats=False,
        momentum=0.9,
    ):
        # Ensure input dimensions are correct
        assert x.shape[-1] == weight.shape[0], "Incompatible dimension between x and weight"

        # Prepare output tensor
        y = torch.empty_like(x)
        # Extract features and batch dimensions
        features = x.shape[-1]
        if x.ndim > 2:
            x = x.view(-1, x.shape[-1])
        M = x.shape[0]

        # Compute number of blocks and launch forward kernel
        BLOCK_SIZE = triton.next_power_of_2(features)
        num_warps = 4
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        rmsnorm_fwd_kernel[M, 1](
            x,
            y,
            weight,
            rstd,
            x.stride(0),
            x.stride(1),
            M,
            features,
            eps,
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.eps = eps
        ctx.features = features
        ctx.x_shape = x.shape
        ctx.y = y
        ctx.rstd = rstd
        ctx.weight = weight
        if track_running_stats:
            ctx.momentum = momentum

        return y

    @staticmethod
    def backward(ctx, dy):
        x = torch.empty_like(dy, dtype=torch.float32)
        x[:, :] = ctx.x.reshape(-1, ctx.features)
        y = ctx.y
        rstd = ctx.rstd
        weight = ctx.weight
        features = ctx.features
        eps = ctx.eps
        M = x.shape[0]

        # Prepare output gradients tensor
        if dy.ndim > 2:
            dy = dy.view(-1, dy.shape[-1])
        grad_input = torch.empty_like(x)
        grad_weight_accum = torch.empty_like(weight)

        # Launch backward kernels
        BLOCK_SIZE = triton.next_power_of_2(features)
        num_warps = 4
        rmsnorm_bwd_kernel[M, 1](
            x,
            weight,
            dy,
            x.stride(0),
            grad_input,
            grad_weight_accum,
            dy.numel(),
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        grad_weight = grad_weight_accum * 0
        return grad_input, grad_weight, None, None, None


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, use_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.use_scale = use_scale
        self.weight = nn.Parameter(torch.ones(dim))
        self.rmsnorm = RMSNormFunction.apply

    def forward(self, x, weight=None, bias=None):
        return self.rmsnorm(x, self.weight, self.eps)
