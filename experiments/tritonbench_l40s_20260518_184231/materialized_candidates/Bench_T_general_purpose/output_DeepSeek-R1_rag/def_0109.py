import torch
import triton
import triton.language as tl

@triton.jit
def compute_sigma_plus_kernel(
    sigma_plus_ptr,
    s_ptr,
    rcond,
    s_row_stride,
    sigma_plus_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    s_row_start = s_ptr + row_idx * s_row_stride
    sigma_plus_row_start = sigma_plus_ptr + row_idx * sigma_plus_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    s_vals = tl.load(s_row_start + col_offsets, mask=mask, other=0.0)
    max_s = tl.max(tl.where(mask, s_vals, -float('inf')), axis=0)
    cutoff = rcond * max_s

    sigma_plus = tl.where(s_vals > cutoff, 1.0 / s_vals, 0.0)
    tl.store(sigma_plus_row_start + col_offsets, sigma_plus, mask=mask)

def pseudoinverse_svd(A, *, full_matrices=True, rcond=1e-15, out=None):
    if A.numel() == 0:
        raise ValueError("Input tensor must not be empty.")

    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)

    original_shape = S.shape
    S_flat = S.reshape(-1, original_shape[-1])
    k = S_flat.shape[-1]
    Sigma_plus_flat = torch.empty_like(S_flat)

    BLOCK_SIZE = triton.next_power_of_2(k)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    compute_sigma_plus_kernel[(S_flat.shape[0],)](
        Sigma_plus_flat,
        S_flat,
        rcond,
        S_flat.stride(0),
        Sigma_plus_flat.stride(0),
        k,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    Sigma_plus = Sigma_plus_flat.reshape(original_shape)
    Vh_scaled = Vh * Sigma_plus.unsqueeze(-2)
    result = Vh_scaled @ U.mH

    if out is not None:
        out.copy_(result)
        return out
    return result
