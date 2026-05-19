import torch
import triton
import triton.language as tl
from torch.autograd.function import Function
from torch.autograd import register_function


class _FusedIndexSelectEq(Function):
    @staticmethod
    def forward(ctx, A, dim, index, other):
        assert dim >= -A.ndim and dim < A.ndim, "Invalid dim"
        assert index.ndim <= 1, "Index should have dimension 1 or 0"
        assert all((i >= 0 and i < A.size(dim)) for i in index), "Index out of range"

        ctx.dim = dim
        ctx.save_for_backward(index)
        dim = dim % A.ndim
        A_shape = list(A.shape)
        index_len = index.numel()
        A = dim_compress(A, dim)
        N = A_shape[dim]
        M = A.numel() // N
        out_shape = list(A.shape)
        out_shape[out_shape.index(N)] = index_len
        out = torch.empty(out_shape, dtype=A.dtype, device=A.device)
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(index_len, META["BLOCK_N"]),
        )

        index_select_kernel[grid](
            A,
            out,
            M,
            N,
            index,
            index_len,
            BLOCK_M=16,
            BLOCK_N=32,
        )
        if isinstance(other, float) or isinstance(other, int):
            other = torch.tensor([other], dtype=A.dtype, device="cuda")
        ret = (out == other.to(out)).all().item()
        if ret:
            return torch.full(out_shape, True, dtype=torch.bool, device="cuda")
        else:
            ctx.out = out
            return out


@register_function("fused_index_select_eq", force=True)
def fused_index_select_eq(A, dim, index, other):
    """
    Fused index select equal function
    """
    return _FusedIndexSelectEq.apply(A, dim, index, other)
