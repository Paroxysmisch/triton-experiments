import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'HAS_SCALE': False}, num_warps=1),
        triton.Config({'HAS_SCALE': True}, num_warps=1),
        triton.Config({'HAS_SCALE': False}, num_warps=2),
        triton.Config({'HAS_SCALE': True}, num_warps=2),
        triton.Config({'HAS_SCALE': False}, num_warps=4),
        triton.Config({'HAS_SCALE': True}, num_warps=4),
        triton.Config({'HAS_SCALE': False}, num_warps=8),
        triton.Config({'HAS_SCALE': True}, num_warps=8),
        triton.Config({'HAS_SCALE': False}, num_warps=16),
        triton.Config({'HAS_SCALE': True}, num_warps=16),
        triton.Config({'HAS_SCALE': False}, num_warps=32),
        triton.Config({'HAS_SCALE': True}, num_warps=32),
    ],
    key=['D', 'B'],
)
@triton.jit
def logsumexp_fwd_kernel(
    x,
    z,
    D,
    B,
    stride_x_n,
    stride_x_d,
    stride_z_n,
    stride_z_d,
    HAS_SCALE: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_D_MAX: tl.constexpr,
):
    i_n = tl.program_id(0)
    i_d = tl.program_id(1)
    o_d = i_d * BLOCK_D_MAX
    m_d = tl.arange(0, BLOCK_D_MAX)

    x_block_ptr = tl.make_block_ptr(
        base=x,
        shape=(D,),
        strides=(stride_x_d,),
        offsets=(i_n * D + o_d + m_d,),
        block_shape=(BLOCK_D_MAX,),
        order=(1,),
    )
    x_b = tl.load(x_block_ptr, boundary_check=(0,), padding_option='zero')

    if HAS_SCALE:
        x_b = (x_b - scale) / scale

    b_m = tl.max(x_b, axis=0)
    x_b = tl.exp(x_b - b_m)
    s_b = tl.sum(x_b, axis=0)
    z_block_ptr = tl.make_block_ptr(
        base=z,
        shape=(D,),
        strides=(stride_z_d,),
        offsets=(i_n * D + o_d + m_d,),
        block_shape=(BLOCK_D_MAX,),
        order=(1,),
    )
    tl.store(z_block_ptr, b_m + tl.log(s_b), boundary_check=(0,))

def logsumexp_fwd(x, dim=-1, scale=None):
    if dim != -1:
        x = x.transpose(dim, -1).contiguous()

    shape = x.shape
    N = torch.prod(torch.tensor(shape[:-1])).item()
    D = x.shape[-1]
    B = triton.next_power_of_2(D)
    BLOCK_D_MAX = triton.next_power_of_2(D // 2)
    x = x.contiguous()
    z = torch.empty_like(x)
    ND = D

    grid = (N, triton.cdiv(D, BLOCK_D_MAX))

    logsumexp_fwd_kernel[grid](
        x,
        z,
        D,
        B,
        x.stride(0),
        x.stride(1),
        z.stride(0),
        z.stride(1),
        scale=scale,
        BLOCK_D_MAX=BLOCK_D_MAX,
    )

    z = z.contiguous()
    z = z.view(N, D)
    z = torch.logsumexp(z, dim=-1)

    return z.to(x.dtype)

def logsumexp(x, dim=-1, keepdim=False, return_log=False):
    if return_log:
        raise ValueError("return_log is not supported")

    smax_value = torch.finfo(x.dtype).min
    x_ = x.view(-1, x.shape[-1])
    dim = x_.dim() - 1
    x_ = x_.transpose(0, dim)
    x_ = x_.contiguous()
    x_max = torch.max(x_, dim=dim, keepdim=True).values
    x_ = torch.log(torch.sum(torch.exp(x_ - x_max), dim=dim, keepdim=True)) + x_max
    if not keepdim:
        x_ = x_.squeeze(dim=dim)

    return x_

def logsumexp_bwd(x, dim=-1):
    s, o = torch.max(x, dim=dim, keepdim=True)
    x = torch.exp(x - s)
    s = torch.sum(x, dim=dim, keepdim=True)
    x = x / s

    if dim != -1:
        x = x.transpose(dim, -1).contiguous()

    return x
