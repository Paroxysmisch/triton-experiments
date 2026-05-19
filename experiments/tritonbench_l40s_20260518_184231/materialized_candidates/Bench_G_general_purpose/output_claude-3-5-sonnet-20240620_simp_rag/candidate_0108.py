import torch
import triton
import triton.language as tl
import math

# Define constants
THETA = 1.0  # You can adjust this value as needed
BLOCK_SIZE_M = 2
BLOCK_SIZE_K = 1024

@triton.jit
def rbe_kernel(
    x_ptr, out_ptr,
    M, K,
    stride_xm, stride_xk,
    stride_om, stride_ok,
    pos,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Compute program ID
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # Compute offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    mask_m = offs_m < M
    mask_k = offs_k < K

    # Compute input and output pointers
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok

    # Load input
    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

    # Compute position-dependent transformation
    pos_k = pos + offs_k
    cos_term = tl.cos(THETA * tl.log(pos_k + 1))
    sin_term = tl.sin(THETA * tl.log(pos_k + 1))

    # Apply transformation
    real_part = x * cos_term
    imag_part = x * sin_term
    result = tl.math.hypot(real_part, imag_part)

    # Store output
    tl.store(out_ptrs, result, mask=mask_m[:, None] & mask_k[None, :])

def rbe_triton_wrapper(x: torch.Tensor, pos: int) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous()
    batch, M, K = x.shape
    out = torch.empty_like(x)

    def grid(meta):
        return (
            triton.cdiv(M, BLOCK_SIZE_M),
            triton.cdiv(K, BLOCK_SIZE_K),
            batch
        )

    rbe_kernel[grid](
        x, out,
        M, K,
        x.stride(1), x.stride(2),
        out.stride(1), out.stride(2),
        pos,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    return out

# Example usage
if __name__ == "__main__":
    batch, M, K = 32, 256, 1024
    x = torch.randn(batch, M, K, device='cuda')
    pos = 0
    result = rbe_triton_wrapper(x, pos)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {result.shape}")
