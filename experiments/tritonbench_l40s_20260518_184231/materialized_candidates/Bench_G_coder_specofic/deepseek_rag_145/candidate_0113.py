import torch
import triton
import triton.language as tl

@triton.jit
def normalize(X, Mean, Var, stride, N, **META):
    """
    Fused layernorm kernel over a 3d tensor.
    The layer norm is applied over the last dimension.

    Compute
        y = (x - E(x))/(sqrt(var(x) + epsilon)) * gamma + beta
    """
    row = tl.program_id(0)
    cols = tl.arange(0, META["BLOCK_SIZE_N"])

    # Move to this row
    x_ptrs = X + row * stride + cols
    x = tl.load(x_ptrs, mask=cols < N, other=0.0).to(tl.float32)
    x = tl.where(cols < N, x, 0.0)

    # Compute mean and variance
    x_mean = tl.sum(x, axis=0) / N
    x_zm = x - x_mean
    x_zm = tl.where(cols < N, x_zm, 0.0)
    x_var = tl.sum(x_zm * x_zm, axis=0) / N
    tl.store(Mean + row, x_mean)
    tl.store(Var + row, x_var)

    # Perform normalization operation
    x_std = tl.sqrt(x_var)
    y = (x - x_mean) / x_std

    return y

def layer_norm(x: torch.Tensor):
    # reshape input data into 2D tensor
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    # heuristics for number of warps.
    num_warps = min(max(BLOCK_SIZE_N // 256, 1), 8)

    mean = torch.zeros((M,)).cuda()
    var = torch.zeros((M,)).cuda()

    # enqueue kernel
    normalize[(M,)](
        x_arg, mean, var,
        x_arg.stride(0),
        N,
        num_warps=num_warps,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )

    return x.reshape(-1, N)
