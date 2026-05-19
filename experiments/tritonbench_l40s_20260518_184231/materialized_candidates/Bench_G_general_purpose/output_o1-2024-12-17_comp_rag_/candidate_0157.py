@triton.jit
def rmsnorm_triton(
    x_ptr,        # *f32
    rms_w_ptr,    # *f32
    output_ptr,   # *f32
    stride_x_batch, stride_x_m, stride_x_n,
    stride_w,     # strides for weight
    stride_o_batch, stride_o_m, stride_o_n,
    N_SIZE,       # total feature size in the last dimension
    eps: tl.float32,
    BLOCK_N_SIZE: tl.constexpr
):
    # Identify which (batch, m) this program instance is responsible for
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Compute base pointers for this batch and row
    x_row_ptr = x_ptr + pid_batch * stride_x_batch + pid_m * stride_x_m
    out_row_ptr = output_ptr + pid_batch * stride_o_batch + pid_m * stride_o_m

    # First pass: compute sum of squares
    sum_squares = tl.zeros([1], dtype=tl.float32)
    for n_start in range(0, N_SIZE, BLOCK_N_SIZE):
        offsets = tl.arange(0, BLOCK_N_SIZE)
        idx = n_start + offsets
        mask = idx < N_SIZE

        x_vals = tl.load(x_row_ptr + idx * stride_x_n, mask=mask, other=0.0)
        sum_squares += tl.sum(x_vals * x_vals, where=mask)

    # Compute rstd = 1 / sqrt(mean_of_squares + eps)
    mean_of_squares = sum_squares / N_SIZE
    rstd = 1.0 / tl.sqrt(mean_of_squares + eps)

    # Second pass: normalize and write output
    for n_start in range(0, N_SIZE, BLOCK_N_SIZE):
        offsets = tl.arange(0, BLOCK_N_SIZE)
        idx = n_start + offsets
        mask = idx < N_SIZE

        x_vals = tl.load(x_row_ptr + idx * stride_x_n, mask=mask, other=0.0)
        w_vals = tl.load(rms_w_ptr + idx, mask=mask, other=1.0)

        norm_vals = x_vals * rstd * w_vals
        tl.store(out_row_ptr + idx * stride_o_n, norm_vals, mask=mask)

def rmsnorm_triton_wrapper(x: torch.Tensor, rms_w: torch.Tensor, eps=1e-5, block_size=128):
    """
    x: 3D tensor of shape [batch_size, M, N_SIZE]
    rms_w: 1D tensor of shape [N_SIZE]
    """
    assert x.is_cuda and rms_w.is_cuda, "Tensors must be on CUDA."
    assert x.dim() == 3, "Input x must have 3 dimensions [batch_size, M, N_SIZE]."
    assert rms_w.dim() == 1, "Weight array must be 1D."
    assert x.size(-1) == rms_w.size(0), "The last dimension of x must match the size of rms_w."

    batch_size, M, N_SIZE = x.shape
    # Allocate output tensor
    out = torch.empty_like(x)

    # Launch kernel with a 2D grid: [batch_size, M]
    rmsnorm_triton[(batch_size, M)](
        x, rms_w, out,
        x.stride(0), x.stride(1), x.stride(2),
        rms_w.stride(0),
        out.stride(0), out.stride(1), out.stride(2),
        N_SIZE,
        eps,
        BLOCK_N_SIZE=block_size
    )

    return out
