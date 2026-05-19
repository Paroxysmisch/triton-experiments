import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({"BT": 32, "BS": 32}, num_warps=4),
        triton.Config({"BT": 64, "BS": 32}, num_warps=8),
        triton.Config({"BT": 128, "BS": 32}, num_warps=16),
        triton.Config({"BT": 256, "BS": 32}, num_warps=32),
    ],
    key=['S'],
)
@triton.jit
def chunk_global_cumsum_vector_kernel(s_ptr, z_ptr, B, H, T, S, stride_sB, stride_sH, stride_sT, stride_sS, stride_zB, stride_zH, stride_zT, stride_zS, BT: tl.constexpr, BS: tl.constexpr):
    pid = tl.program_id(axis=0)
    bid = pid // H
    hid = pid % H

    s_ptr += bid * stride_sB + hid * stride_sH
    z_ptr += bid * stride_zB + hid * stride_zH

    offs_t = tl.arange(0, BT)
    offs_s = tl.arange(0, BS)
    
    mask_t = offs_t < T
    mask_s = offs_s < S

    for t in range(0, T, BT):
        t_offset = t + offs_t
        mask = mask_t & (t_offset < T)
        
        s_block_ptr = tl.make_block_ptr(s_ptr + t_offset[:, None] * stride_sT + offs_s[None, :] * stride_sS, shape=(T, S), strides=(stride_sT, stride_sS), offsets=(0, 0), block_shape=(BT, BS), order=(0, 1))
        b_s = tl.load(s_block_ptr, mask=mask[:, None] & mask_s[None, :], other=0.0).to(tl.float32)
        
        m_s = tl.zeros((BT, BT), dtype=tl.float32)
        for i in range(BT):
            m_s += tl.eye(BT, k=-i, dtype=tl.float32)
        
        b_c = tl.dot(m_s, b_s)
        
        z_block_ptr = tl.make_block_ptr(z_ptr + t_offset[:, None] * stride_zT + offs_s[None, :] * stride_zS, shape=(T, S), strides=(stride_zT, stride_zS), offsets=(0, 0), block_shape=(BT, BS), order=(0, 1))
        tl.store(z_block_ptr, b_c, mask=mask[:, None] & mask_s[None, :])

def chunk_global_cumsum_vector(s):
    B, H, T, S = s.shape
    z = torch.empty_like(s, dtype=torch.float32)
    
    grid = lambda meta: (B * H, )
    
    chunk_global_cumsum_vector_kernel[grid](s, z, B, H, T, S, s.stride(0), s.stride(1), s.stride(2), s.stride(3), z.stride(0), z.stride(1), z.stride(2), z.stride(3))
    
    return z

# Example usage:
s = torch.randn(2, 3, 64, 128, device='cuda', dtype=torch.float32)
z = chunk_global_cumsum_vector(s)
print(z)
