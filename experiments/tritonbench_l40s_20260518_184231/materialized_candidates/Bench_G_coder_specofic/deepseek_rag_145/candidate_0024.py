import torch
import triton
import triton.language as tl

@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel(s, o, T: tl.constexpr):
    i_bh = tl.program_id(0)
    b_z = tl.zeros((T,), dtype=tl.float32)
    for i_t in range(T - 1, -1, -1):
        b_s = s[i_bh, i_t]
        b_z[i_t] = b_z[i_t] + b_s
        o[i_bh, i_t] = b_z[i_t]

def chunk_global_reversed_cumsum_scalar(s: torch.Tensor):
    B, H, T = s.shape
    grid = (B * H,)
    o = torch.empty_like(s)
    chunk_global_reversed_cumsum_scalar_kernel[grid](s, o, T=T)
    return o
