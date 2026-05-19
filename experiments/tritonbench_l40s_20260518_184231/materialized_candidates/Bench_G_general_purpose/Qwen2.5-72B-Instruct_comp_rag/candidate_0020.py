import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BT': 16}, num_warps=2),
        triton.Config({'BT': 16}, num_warps=4),
        triton.Config({'BT': 16}, num_warps=8),
        triton.Config({'BT': 32}, num_warps=2),
        triton.Config({'BT': 32}, num_warps=4),
        triton.Config({'BT': 32}, num_warps=8),
        triton.Config({'BT': 64}, num_warps=2),
        triton.Config({'BT': 64}, num_warps=4),
        triton.Config({'BT': 64}, num_warps=8),
    ],
    key=['T']
)
@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel(
    s,
    o,
    s_s_t,
    s_s_h,
    T: tl.constexpr,
    BT: tl.constexpr
):
    i_h = tl.program_id(0)
    b_z = tl.zeros([1], dtype=tl.float32)
    
    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        p_s = tl.make_block_ptr(s + i_h * s_s_h, (T,), (s_s_t,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + i_h * s_s_h, (T,), (s_s_t,), (i_t * BT,), (BT,), (0,))
        
        b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
        b_c = b_z + b_s
        tl.store(p_o, b_c.to(p_o.dtype.element_ty), boundary_check=(0,))
        
        b_z += tl.sum(b_s, 0)

def chunk_global_reversed_cumsum_scalar(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    B, H, T = s.shape
    BT = 32

    dtype = dtype or s.dtype
    grid = (B * H,)
    o = torch.empty_like(s, dtype=dtype)
    chunk_global_reversed_cumsum_scalar_kernel[grid](
        s, o,
        s.stride(2), s.stride(1),
        T=T, BT=BT
    )
    return o

class ReversedCumsumFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, s, dtype):
        o = chunk_global_reversed_cumsum_scalar(s, dtype)
        ctx.dtype = dtype
        return o

    @staticmethod
    def backward(ctx, do):
        ds = chunk_global_reversed_cumsum_scalar(do, ctx.dtype)
        return ds, None

def reversed_cumsum(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    return ReversedCumsumFunction.apply(s, dtype)
