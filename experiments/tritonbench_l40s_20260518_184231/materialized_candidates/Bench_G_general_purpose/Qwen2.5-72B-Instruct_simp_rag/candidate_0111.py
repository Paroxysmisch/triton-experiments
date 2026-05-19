import torch
import triton
import triton.language as tl

# Triton kernel for fused layer normalization
@triton.jit
def triton_red_fused_native_layer_norm_0(primals_1, primals_2, primals_3, buf0, buf3, buf4, stride, D, epsilon, **META):
    """
    Fused layernorm kernel over a 2D tensor.
    The layer norm is applied over the last dimension.

    Compute
        y = (x - E(x))/(sqrt(var(x) + epsilon)) * gamma + beta
    """

    row = tl.program_id(0)
    cols = tl.arange(0, META["BLOCK_SIZE_D"])

    # Move to this row
    x_ptrs = primals_3 + row * stride + cols
    x = tl.load(x_ptrs, mask=cols < D, other=0.0).to(tl.float32)
    x = tl.where(cols < D, x, 0.0)

    # Compute mean
    x_mean = tl.sum(x, axis=0) / D
    x_zm = x - x_mean
    x_zm = tl.where(cols < D, x_zm, 0.0)

    # Compute variance
    x_var = tl.sum(x_zm * x_zm, axis=0) / D
    x_var_inv_sqrt = 1.0 / tl.sqrt(x_var + epsilon)

    # Normalize
    y = x_zm * x_var_inv_sqrt
    y = tl.where(cols < D, y, 0.0)

    # Apply gamma and beta
    gamma = tl.load(primals_1 + cols, mask=cols < D, other=1.0).to(tl.float32)
    beta = tl.load(primals_2 + cols, mask=cols < D, other=0.0).to(tl.float32)
    y = y * gamma + beta
    y = tl.where(cols < D, y, 0.0)

    # Store results
    tl.store(buf0 + row, x_mean)
    tl.store(buf3 + row, x_var)
    tl.store(buf4 + row * stride + cols, y, mask=cols < D)

# Wrapper function for the fused layer normalization
def fused_native_layer_norm(primals_1, primals_2, primals_3):
    S, D = primals_3.shape

    MAX_FUSED_SIZE = 65536 // primals_3.element_size()
    BLOCK_SIZE_D = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BLOCK_SIZE_D:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    # heuristics for number of warps.
    num_warps = min(max(BLOCK_SIZE_D // 256, 1), 8)

    buf0 = torch.zeros((S,)).cuda()
    buf3 = torch.zeros((S,)).cuda()
    buf4 = torch.zeros_like(primals_3).cuda()

    # enqueue kernel
    triton_red_fused_native_layer_norm_0[(S,)](
        primals_1, primals_2, primals_3, buf0, buf3, buf4,
        primals_3.stride(0),
        D,
        1e-5,  # epsilon
        num_warps=num_warps,
        BLOCK_SIZE_D=BLOCK_SIZE_D
    )

    return buf4, primals_3, buf0, buf3
