import torch
import triton
import triton.language as tl

@triton.jit
def rbe_triton(
    X, stride_x_batch, stride_xm, stride_xk,
    Z, stride_z_batch, stride_zm, stride_zk,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    M, K
):
    batch_idx = tl.program_id(0)
    m_block_idx = tl.program_id(1)
    k_block_idx = tl.program_id(2)

    offs_batch = batch_idx
    offs_m = m_block_idx * BLOCK_SIZE_M
    offs_k = k_block_idx * BLOCK_SIZE_K

    m_idx = tl.arange(0, BLOCK_SIZE_M)
    k_real_idx = tl.arange(0, BLOCK_SIZE_K // 2)
    m_offs = offs_m + m_idx[:, None]
    k_real_offs = offs_k + 2 * k_real_idx[None, :]

    mask_m = (m_offs < M)
    mask_k = (k_real_offs < K)
    mask = mask_m & mask_k

    x_real_ptrs = X + offs_batch * stride_x_batch + m_offs * stride_xm + k_real_offs * stride_xk
    real = tl.load(x_real_ptrs, mask=mask, other=0.0)
    x_imag_ptrs = X + offs_batch * stride_x_batch + m_offs * stride_xm + (k_real_offs + 1) * stride_xk
    imag = tl.load(x_imag_ptrs, mask=mask, other=0.0)

    i_global = (offs_k // 2) + k_real_idx[None, :]
    d = K // 2
    theta = 10000.0
    inv_freq = 1.0 / (theta ** (2.0 * i_global / tl.cast(d, tl.float32)))
    angle = tl.cast(m_offs, tl.float32) * inv_freq
    cos_vals = tl.cos(angle)
    sin_vals = tl.sin(angle)

    out_real = real * cos_vals - imag * sin_vals
    out_imag = real * sin_vals + imag * cos_vals

    z_real_ptrs = Z + offs_batch * stride_z_batch + m_offs * stride_zm + k_real_offs * stride_zk
    z_imag_ptrs = Z + offs_batch * stride_z_batch + m_offs * stride_zm + (k_real_offs + 1) * stride_zk
    tl.store(z_real_ptrs, out_real, mask=mask)
    tl.store(z_imag_ptrs, out_imag, mask=mask)

def rbe_triton_wrapper(x: torch.Tensor) -> torch.Tensor:
    batch, M, K = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE_M = 2
    BLOCK_SIZE_K = 1024

    grid_batch = batch
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_k = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    rbe_triton[(grid_batch, grid_m, grid_k)](
        x, x.stride(0), x.stride(1), x.stride(2),
        out, out.stride(0), out.stride(1), out.stride(2),
        BLOCK_SIZE_M, BLOCK_SIZE_K, M, K
    )
    return out

# Example usage:
# x = torch.randn(batch, M, K, dtype=torch.float32, device='cuda')
# out = rbe_triton_wrapper(x)
