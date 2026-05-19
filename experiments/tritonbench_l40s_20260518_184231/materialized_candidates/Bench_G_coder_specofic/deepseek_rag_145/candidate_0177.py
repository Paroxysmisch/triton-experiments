import triton
import triton.language as tl

@triton.autotune(key=["S"], configs=[triton.Config({"BT":64, "BS":64})])
@triton.jit
def chunk_global_cumsum_vector_kernel(s_ptr, z_ptr, B, T, S,
                                     stride_batch, stride_head, stride_time, stride_size,
                                     BT: tl.constexpr, BS: tl.constexpr):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_t = tl.program_id(axis=2)
    pid_s = tl.program_id(axis=3)

    block_start_time = pid_t * BT
    m_s = block_start_time <= tl.arange(0, BS)

    s_ptr = s_ptr + pid_b * stride_batch + pid_h * stride_head + block_start_time * stride_time + pid_s * stride_size
    z_ptr = z_ptr + pid_b * stride_batch + pid_h * stride_head + block_start_time * stride_time + pid_s * stride_size
    
    b_s = tl.load(s_ptr, mask=m_s, other=0.0).to(tl.float32)
    b_z = tl.load(z_ptr, mask=m_s & (block_start_time>0), other=0.0).to(tl.float32)

    b_c = tl.dot(b_s, m_s)
    b_z_next = b_z + b_c
    tl.store(z_ptr, b_z_next, mask=m_s)

    return b_z_next

def chunk_global_cumsum_vector(s, B, T, S, BT=64, BS=64):
    assert s.shape == (B, T, S)
    Z = torch.zeros_like(s, dtype=torch.float32)
    grid = lambda meta: (B, T // BT, S, triton.cdiv(meta['BT'], BS))
    chunk_global_cumsum_vector_kernel[grid](
        s, Z, B, T, S,
        s.stride(0), s.stride(1), s.stride(2), 1,
        BT, BS
    )
    return Z
