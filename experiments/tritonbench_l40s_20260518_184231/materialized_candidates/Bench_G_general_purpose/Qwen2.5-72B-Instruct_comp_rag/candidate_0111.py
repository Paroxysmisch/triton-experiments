import torch
import triton
import triton.language as tl

@triton.jit
def k_layer_norm(X, Mean, Var, Out, Scale, Bias, stride, N, epsilon, **META):
    """
    Fused layer normalization kernel over a 3D tensor.
    The layer norm is applied over the last dimension.

    Compute
        y = (x - E(x)) / (sqrt(var(x) + epsilon)) * gamma + beta
    """

    row = tl.program_id(0)
    cols = tl.arange(0, META["BLOCK_SIZE_N"])

    # Move to this row
    x_ptrs = X + row * stride + cols
    x = tl.load(x_ptrs, mask=cols < N, other=0.0).to(tl.float32)
    x = tl.where(cols < N, x, 0.0)

    # Compute mean
    x_mean = tl.sum(x, axis=0) / N

    # Compute variance
    x_zm = x - x_mean
    x_zm = tl.where(cols < N, x_zm, 0.0)
    x_var = tl.sum(x_zm * x_zm, axis=0) / N

    # Store mean and variance
    tl.store(Mean + row, x_mean)
    tl.store(Var + row, x_var)

    # Compute normalization factor
    norm_factor = 1.0 / tl.sqrt(x_var + epsilon)

    # Apply affine transformation
    scale = tl.load(Scale + cols, mask=cols < N, other=1.0).to(tl.float32)
    bias = tl.load(Bias + cols, mask=cols < N, other=0.0).to(tl.float32)

    # Normalize and store the result
    y = (x - x_mean) * norm_factor * scale + bias
    y = tl.where(cols < N, y, 0.0)
    tl.store(Out + row * stride + cols, y, mask=cols < N)

def layer_norm(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, epsilon: float = 1e-5):
    # Reshape input data into 2D tensor
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape

    # Determine block size
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    # Heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE_N // 256, 1), 8)

    # Prepare output tensors
    mean = torch.zeros((M,)).cuda()
    var = torch.zeros((M,)).cuda()
    out = torch.empty_like(x_arg)

    # Enqueue kernel
    k_layer_norm[(M,)](
        x_arg, mean, var, out, scale, bias,
        x_arg.stride(0),
        N,
        epsilon,
        num_warps=num_warps,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )

    # Reshape output to match input shape
    out = out.reshape(x.shape)

    return out, mean.reshape(x.shape[:-1]), var.reshape(x.shape[:-1])
