import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BT': 64, 'BS': 64}, num_warps=4),
        triton.Config({'BT': 128, 'BS': 64}, num_warps=4),
        triton.Config({'BT': 64, 'BS': 128}, num_warps=4),
        triton.Config({'BT': 128, 'BS': 128}, num_warps=4),
        triton.Config({'BT': 256, 'BS': 64}, num_warps=8),
        triton.Config({'BT': 256, 'BS': 128}, num_warps=8),
    ],
    key=['S'],
)
@triton.jit
def chunk_global_cumsum_vector_kernel(
    s_ptr, z_ptr,
    B, H, T, S,
    stride_s_b, stride_s_h, stride_s_t, stride_s_s,
    stride_z_b, stride_z_h, stride_z_t, stride_z_s,
    BT: tl.constexpr, BS: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_sb = tl.program_id(2)

    off_b = pid_b
    off_h = pid_h
    off_s = pid_sb * BS

    b_z = tl.zeros([BS], dtype=tl.float32)

    for t in range(0, T, BT):
        s_block_ptr = tl.make_block_ptr(
            base=s_ptr + off_b * stride_s_b + off_h * stride_s_h,
            shape=(T, S),
            strides=(stride_s_t, stride_s_s),
            offsets=(t, off_s),
            block_shape=(BT, BS),
            order=(1, 0)
        )

        z_block_ptr = tl.make_block_ptr(
            base=z_ptr + off_b * stride_z_b + off_h * stride_z_h,
            shape=(T, S),
            strides=(stride_z_t, stride_z_s),
            offsets=(t, off_s),
            block_shape=(BT, BS),
            order=(1, 0)
        )

        b_s = tl.load(s_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b_s = b_s.to(tl.float32)

        m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
        m_s = m_s.to(tl.float32)

        b_c = tl.dot(m_s, b_s)
        b_c += b_z[None, :]

        tl.store(z_block_ptr, b_c.to(z_block_ptr.dtype.element_ty))
        b_z = b_c[-1, :]

def chunk_global_cumsum_vector(s: torch.Tensor, BT: int, BS: int):
    B, H, T, S = s.shape
    z = torch.empty_like(s)
    grid = (B, H, (S + BS - 1) // BS)
    chunk_global_cumsum_vector_kernel[grid](
        s, z,
        B, H, T, S,
        s.stride(0), s.stride(1), s.stride(2), s.stride(3),
        z.stride(0), z.stride(1), z.stride(2), z.stride(3),
        BT=BT, BS=BS
    )
    return z

# Example Usage
if __name__ == "__main__":
    B, H, T, S = 2, 3, 128, 64
    s = torch.randn(B, H, T, S, device='cuda')
    output = chunk_global_cumsum_vector(s, BT=64, BS=64)
    print(output.shape)  # Expected: torch.Size([2, 3, 128, 64])
