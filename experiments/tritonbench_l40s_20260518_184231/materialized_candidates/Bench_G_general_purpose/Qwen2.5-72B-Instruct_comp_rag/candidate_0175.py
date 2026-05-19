import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BT": 16, "BS": 16}, num_warps=4),
        triton.Config({"BT": 32, "BS": 32}, num_warps=8),
        triton.Config({"BT": 64, "BS": 64}, num_warps=16),
    ],
    key=["B", "H", "T", "S"]
)
@triton.jit
def chunk_global_cumsum_vector_kernel(
    s_ptr, z_ptr,
    B, H, T, S,
    stride_s_B, stride_s_H, stride_s_T, stride_s_S,
    stride_z_B, stride_z_H, stride_z_T, stride_z_S,
    BT: tl.constexpr, BS: tl.constexpr
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_t = tl.program_id(axis=2)

    # Create a lower triangular mask
    mask = tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :]
    mask = mask.to(tl.float32)

    # Initialize pointers
    s_block_ptr = tl.make_block_ptr(
        base=s_ptr,
        shape=(B, H, T, S),
        strides=(stride_s_B, stride_s_H, stride_s_T, stride_s_S),
        offsets=(pid_b * BT, pid_h * BT, pid_t * BT, 0),
        block_shape=(1, 1, BT, BS),
        order=(3, 0, 1, 2)
    )
    z_block_ptr = tl.make_block_ptr(
        base=z_ptr,
        shape=(B, H, T, S),
        strides=(stride_z_B, stride_z_H, stride_z_T, stride_z_S),
        offsets=(pid_b * BT, pid_h * BT, pid_t * BT, 0),
        block_shape=(1, 1, BT, BS),
        order=(3, 0, 1, 2)
    )

    # Load data
    b_s = tl.load(s_block_ptr, mask=mask, other=0.0).to(tl.float32)
    b_z = tl.zeros((BT, BS), dtype=tl.float32)

    # Compute block-level cumulative sum
    for i in range(BT):
        b_c = tl.dot(mask[i, :i+1], b_s[i, :])
        b_z[i, :] = b_c

    # Store the result
    tl.store(z_block_ptr, b_z, mask=mask)

import torch

def chunk_global_cumsum_vector(s, BT=16, BS=16):
    B, H, T, S = s.shape
    z = torch.empty_like(s, dtype=torch.float32, device=s.device)

    # Define the grid
    grid = (triton.cdiv(B, BT), triton.cdiv(H, BT), triton.cdiv(T, BT))

    # Call the kernel
    chunk_global_cumsum_vector_kernel[grid](
        s, z,
        B, H, T, S,
        s.stride(0), s.stride(1), s.stride(2), s.stride(3),
        z.stride(0), z.stride(1), z.stride(2), z.stride(3),
        BT=BT, BS=BS
    )

    return z
