import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"HAS_SCALE": False}, num_warps=1),
        triton.Config({"HAS_SCALE": False}, num_warps=2),
        triton.Config({"HAS_SCALE": False}, num_warps=4),
        triton.Config({"HAS_SCALE": False}, num_warps=8),
        triton.Config({"HAS_SCALE": False}, num_warps=16),
        triton.Config({"HAS_SCALE": False}, num_warps=32),
        triton.Config({"HAS_SCALE": True}, num_warps=1),
        triton.Config({"HAS_SCALE": True}, num_warps=2),
        triton.Config({"HAS_SCALE": True}, num_warps=4),
        triton.Config({"HAS_SCALE": True}, num_warps=8),
        triton.Config({"HAS_SCALE": True}, num_warps=16),
        triton.Config({"HAS_SCALE": True}, num_warps=32),
    ],
    key=["D"],
)
@triton.jit
def logsumexp_fwd_kernel(
    x,
    z,
    scale,
    N: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_z_n: tl.constexpr,
    stride_z_d: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    i_n = tl.program_id(0)
    i_d = tl.program_id(1)
    o_d = i_d * B
    m_d = o_d + tl.arange(0, B)

    mask = m_d < D

    b_x = tl.load(
        x + i_n * stride_x_n + m_d * stride_x_d, mask=mask, other=-float("inf")
    ).to(tl.float32)
    if HAS_SCALE:
        b_scale = tl.load(scale + m_d * stride_x_d, mask=mask).to(tl.float32)
        b_x *= b_scale
    b_m = tl.max(b_x, axis=0)
    b_x = tl.log(tl.sum(tl.exp(b_x - b_m), axis=0)) + b_m

    z_offset = i_n * stride_z_n + o_d * stride_z_d
    tl.store(z + z_offset + m_d, b_x, mask=m_d < D)


def logsumexp_fwd(x, scale=None, dim=None, keepdim=False, dtype=None):
    if dim is None:
        x = x.flatten()
        dim = -1
    else:
        x = x.contiguous()

    N = x.shape[0]
    D = x.shape[1]
    B = triton.next_power_of_2(D)

    if dtype is None:
        dtype = x.dtype
    z = torch.empty((N, D), device=x.device, dtype=dtype)
    ND = D // B

    grid = (N, ND)

    logsumexp_fwd_kernel[grid](
        x,
        z,
        scale,
        N,
        D,
        B,
        x.stride(0),
        x.stride(1),
        z.stride(0),
        z.stride(1),
        D % B == 0,
    )

    if not keepdim:
        z = torch.squeeze(z, dim)
    return z
