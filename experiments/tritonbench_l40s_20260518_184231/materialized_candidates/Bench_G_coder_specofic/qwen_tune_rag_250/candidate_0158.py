, grad_x, partial_grad_weight, x, weight,
            x.stride(0), H, num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)
        # Weighted sum is a sum, so the total gradient wrt. weight is the partial sum.
        grad_weight = partial_grad_weight.sum(0)
        return grad_x, grad_weight

def weighted_sum_triton(x, weight):
    return WeightedSumFunc_Triton.apply(x, weight)

@triton.jit
def rms_norm_fwd(x_ptr, rms_w_ptr, output_ptr, x_row_stride,
                 M: tl.uint32, N_SIZE: tl.constexpr, eps: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):
    # The rms norm is applied over the last dimension.
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    # The stride represents how much we need to increase the
    # pointer to advance 1 row.
    row_start_ptr = x_ptr + pid_batch * x_row_stride + pid_m * N_SIZE
    # The block size is the next power of two greater than N, so we can
    # fit each row in a single block.
    offsets_n = tl.arange(0, BLOCK_N_SIZE)
    # Create a mask since BLOCK_N_SIZE may be > than N.
    mask = offsets_n < N_SIZE
    # Load the row into SRAM, using a mask since BLOCK_N_SIZE may be > N.
    x = tl.load(row_start_ptr + offsets_n, mask=mask, other=0)
    # The rms norm computes the variance over the last dimension.
    # We compute the variance across the entire block, which allows
    # for greater optimization opportunities, but requires additional
    # computation since we then have to divide by N instead of just
    # taking the square root of the sum.
    variance = tl.sum(x * x, axis=0) / N_SIZE
    rstd = 1 / tl.sqrt(variance + eps)
    # Normalize, optionally affine.
    output = x * rstd
    # Write back output.
    tl.store(output_ptr, output)

class RmsNormTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rms_w, eps):
        ctx.eps = eps
        ctx.BLOCK_N_SIZE = triton.next_power_of_2(x.shape[-1])
        # Save x and rms_w for backward.
        ctx.save_for_backward(x, rms_w)
        M, output_dims = x.shape[:-1]
        assert rms_w.shape[-1] == x.shape[-1], "Incompatible dimensions between x and rms_w"
        # The rms norm is applied over the last dimension, so we reshape x
        # into 2D, with each row representing a single rms norm computation.
        x_reshaped = x.reshape(-1, x.shape[-1])
        N_SIZE, output_dims = x_reshaped.shape
        # Allocate output.
        output = torch.empty_like(x_reshaped)
        # Launch kernel.
        rms_norm_fwd[(M, output_dims,)](
            x_reshaped, rms_w, output, x_reshaped.stride(0),
            M, N_SIZE, eps, BLOCK_N_SIZE=ctx.BLOCK_N_SIZE)
        return output.reshape_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, rms_w = ctx.saved_tensors
        eps = ctx.eps
        N, M = x.shape
        grad_x = torch.empty_like(x)
        partial_grad_rms_w = torch.empty_like(x)
        # Launch kernel.
        rms_norm_backward[(M,)](
            grad_output, grad_x, partial_grad_rms_w, x, rms_w,
            x.stride(0), N, eps, BLOCK_N_SIZE=ctx.BLOCK_N_SIZE)
        # Sum up partial gradients.
        grad_rms_w = partial_grad_rms_w.sum(0)
        return grad_x, grad_rms_w, None

def rms_norm_triton(x, rms_w, eps):
    return RmsNormTriton.apply(x, rms_w, eps)

@triton.jit
def rms_norm_backward(grad_output_ptr, grad_x_ptr, partial_grad_rms_w_ptr, x_ptr, rms_w_ptr, x_row_stride,
                      M: tl.uint32, N_SIZE: tl.constexpr, eps: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):
    # The rms norm is applied over the last dimension.
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    # The stride represents how much we need to increase the
    # pointer to advance 1 row.
    row_start_ptr = x_ptr + pid_batch * x_row_stride + pid_m * N_SIZE
    # The block size is the next power of two greater than N, so we can
    # fit each row in a single block.
    offsets_n = tl.arange(0, BLOCK_N_SIZE)
    # Create a mask since BLOCK_N_SIZE may be > than N.
    mask = offsets_n < N_SIZE
    # Load the row into SRAM, using a mask since BLOCK_N_SIZE may be > N.
    x = tl.load(row_start_ptr + offsets_n, mask=mask, other=0)
    rms_w = tl.load(rms_w_ptr + offsets_n, mask=mask, other=0)
    # Load the grad_output into SRAM.
    grad_output = tl.load(grad_output_ptr + pid_batch * M + pid_m)
    # The rms norm computes the variance over the last dimension.
    # We compute the variance across the entire block, which allows
    # for greater optimization opportunities, but requires additional
    # computation since we then have to divide by N instead of just
    # taking the square root of the sum.
    # Normalize, optionally affine.
    grad_x = (x * rms_w) * grad_output
    # Write out gradients.
    tl.store(grad_x_ptr + row_start_ptr, grad_x, mask=mask)
    partial_grad_rms_w = (rms_w * grad_output) * x
    tl.store(partial_grad_rms_w_ptr + row_start_ptr, partial_grad_rms_w, mask=mask)
