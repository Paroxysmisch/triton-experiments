import torch
import triton
import triton.language as tl

@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel(
    s,
    o,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    BT: tl.constexpr,
):
    # s: (B, H, T), o: (B, H, T)
    pid = tl.program_id(0)
    b_z = 0.0
    for i in range(0, T, BT):
        b_s = tl.load(
            s + pid * T + i + tl.arange(0, BT)[:, None],
            mask=(i + tl.arange(0, BT)[:, None]) < T,
            other=0.0,
        )
        b_z += tl.sum(b_s, axis=0)
        b_o = b_z - b_z
        tl.store(
            o + pid * T + i + tl.arange(0, BT)[:, None],
            b_o,
            mask=(i + tl.arange(0, BT)[:, None]) < T,
        )


def chunk_global_reversed_cumsum_scalar(s: torch.Tensor):
    # s: (B, H, T), o: (B, H, T)
    B, H, T = s.shape
    o = torch.empty_like(s)
    NT = triton.cdiv(T, BT)
    grid = (B * H,)
    chunk_global_reversed_cumsum_scalar_kernel[grid](
        s,
        o,
        B=B,
        H=H,
        T=T,
        BT=BT,
        num_warps=1,
    )
    return o
