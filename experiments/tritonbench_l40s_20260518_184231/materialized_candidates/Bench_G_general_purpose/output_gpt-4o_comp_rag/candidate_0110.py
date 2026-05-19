import torch
import triton
import triton.language as tl

@triton.jit
def triton_red_fused_native_layer_norm_0(primals_3, primals_1, primals_2, out_ptr0, out_ptr1, stride, N, epsilon, **META):
    """
    Fused layer normalization kernel for 2D tensor using Welford's algorithm.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, META["RBLOCK"])

    # Move to this row
    x_ptrs = primals_3 + row * stride + cols
    x = tl.load(x_ptrs, mask=cols < N, other=0.0).to(tl.float32)
    x = tl.where(cols < N, x, 0.0)

    # Welford algorithm for mean and variance
    mean = tl.zeros([1], dtype=tl.float32)
    m2 = tl.zeros([1], dtype=tl.float32)
    count = tl.zeros([1], dtype=tl.float32)

    for i in range(0, N, META["RBLOCK"]):
        x_i = tl.load(x_ptrs + i, mask=cols < N, other=0.0).to(tl.float32)
        x_i = tl.where(cols < N, x_i, 0.0)
        delta = x_i - mean
        count += 1
        mean += delta / count
        delta2 = x_i - mean
        m2 += delta * delta2

    variance = m2 / count
    inv_std = 1.0 / tl.sqrt(variance + epsilon)

    # Store intermediate results
    tl.store(out_ptr0 + row * 3 + 0, mean)
    tl.store(out_ptr0 + row * 3 + 1, variance)
    tl.store(out_ptr0 + row * 3 + 2, count)

    # Normalize the input
    norm_x = (x - mean) * inv_std

    # Apply affine transformation
    if primals_1 is not None and primals_2 is not None:
        gamma = tl.load(primals_1 + cols, mask=cols < N, other=1.0)
        beta = tl.load(primals_2 + cols, mask=cols < N, other=0.0)
        norm_x = norm_x * gamma + beta

    # Store the final result
    tl.store(out_ptr1 + row * stride + cols, norm_x, mask=cols < N)

def fused_native_layer_norm(primals_3, primals_1=None, primals_2=None, epsilon=1e-5):
    # Ensure input is a 2D tensor
    assert primals_3.ndim == 2
    S, D = primals_3.shape

    # Allocate output buffers
    buf0 = torch.zeros((S, 3), device=primals_3.device, dtype=torch.float32)
    buf4 = torch.zeros_like(primals_3, device=primals_3.device, dtype=torch.float32)

    # Determine block size
    MAX_FUSED_SIZE = 65536 // primals_3.element_size()
    RBLOCK = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > RBLOCK:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    # Heuristics for number of warps
    num_warps = min(max(RBLOCK // 256, 1), 8)

    # Enqueue kernel
    triton_red_fused_native_layer_norm_0[(S,)](
        primals_3, primals_1, primals_2, buf0, buf4,
        primals_3.stride(0),
        D,
        epsilon,
        num_warps=num_warps,
        RBLOCK=RBLOCK
    )

    # Extract mean and variance from buffer
    mean = buf0[:, 0]
    variance = buf0[:, 1]

    return buf4, mean, variance

# Example usage
if __name__ == "__main__":
    S, D = 1024, 512
    primals_3 = torch.randn((S, D), device='cuda', dtype=torch.float32)
    primals_1 = torch.randn((D,), device='cuda', dtype=torch.float32)
    primals_2 = torch.randn((D,), device='cuda', dtype=torch.float32)

    out, mean, variance = fused_native_layer_norm(primals_3, primals_1, primals_2)
    print(out)
    print(mean)
    print(variance)
