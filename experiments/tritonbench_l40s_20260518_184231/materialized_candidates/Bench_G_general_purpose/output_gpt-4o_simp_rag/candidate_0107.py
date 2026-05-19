import torch
import triton
import triton.language as tl

@triton.jit
def triton_red_fused_native_layer_norm_0(X, Gamma, Beta, Buf0, Buf3, Buf4, stride, D, epsilon, **META):
    """
    Triton kernel for fused layer normalization over a 2D tensor.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, META["BLOCK_SIZE_D"])

    # Load the row of the input tensor
    x_ptrs = X + row * stride + cols
    x = tl.load(x_ptrs, mask=cols < D, other=0.0).to(tl.float32)

    # Compute mean
    x_mean = tl.sum(x, axis=0) / D
    tl.store(Buf0 + row, x_mean)

    # Compute variance
    x_zm = x - x_mean
    x_var = tl.sum(x_zm * x_zm, axis=0) / D
    tl.store(Buf3 + row, x_var)

    # Normalize
    x_norm = (x - x_mean) / tl.sqrt(x_var + epsilon)

    # Apply scale and shift
    gamma_ptrs = Gamma + cols
    beta_ptrs = Beta + cols
    gamma = tl.load(gamma_ptrs, mask=cols < D, other=1.0)
    beta = tl.load(beta_ptrs, mask=cols < D, other=0.0)

    y = x_norm * gamma + beta
    tl.store(Buf4 + row * stride + cols, y, mask=cols < D)

def fused_native_layer_norm(primals_1, primals_2, primals_3, epsilon=1e-5):
    # Reshape input tensor into 2D tensor
    S, D = primals_3.shape
    x_arg = primals_3.view(S, D)

    # Determine block size
    MAX_FUSED_SIZE = 65536 // primals_3.element_size()
    BLOCK_SIZE_D = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BLOCK_SIZE_D:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    # Heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE_D // 256, 1), 8)

    # Allocate buffers
    buf0 = torch.empty((S,), device='cuda', dtype=torch.float32)
    buf3 = torch.empty((S,), device='cuda', dtype=torch.float32)
    buf4 = torch.empty_like(primals_3)

    # Enqueue kernel
    triton_red_fused_native_layer_norm_0[(S,)](
        x_arg, primals_1, primals_2, buf0, buf3, buf4,
        x_arg.stride(0),
        D,
        epsilon,
        num_warps=num_warps,
        BLOCK_SIZE_D=BLOCK_SIZE_D
    )

    return buf4, primals_3, buf0, buf3
