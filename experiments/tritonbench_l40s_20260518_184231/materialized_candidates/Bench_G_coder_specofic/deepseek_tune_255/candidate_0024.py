import torch
import triton
import triton.language as tl

@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel(s, o, BT: tl.constexpr):
    # Kernel to compute reversed cumulative sum
    b_z = tl.zeros([BT], dtype=tl.float32)
    for i in range(0, BT):
        tl.device_assert(i < BT, "i is out of bounds")
        b_z += tl.load(s + i)
    b_z = tl.sum(b_z, axis=0)
    for i in range(BT - 1, -1, -1):
        tl.device_assert(i < BT, "i is out of bounds")
        b_z = tl.where(i < BT - 1, b_z - tl.load(s + i), b_z)
        tl.store(o + i, b_z)

def chunk_global_reversed_cumsum_scalar(s: torch.Tensor):
    # Wrapper function to compute reversed cumulative sum
    o = torch.zeros_like(s, dtype=torch.float32)
    BT = s.shape[-1]
    grid = lambda meta: (triton.cdiv(1, meta["BT"]),)
    chunk_global_reversed_cumsum_scalar_kernel[grid](s, o, BT=BT)
    return o
