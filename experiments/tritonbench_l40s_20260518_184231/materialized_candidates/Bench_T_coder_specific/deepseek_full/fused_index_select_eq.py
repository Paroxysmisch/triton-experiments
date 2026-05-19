import torch
import triton
import triton.language as tl

@triton.jit
def fused_index_select_eq(input, dim, index, other, out=None):
    # index selection
    N = input.shape[dim]
    M = index.shape[0]
    K = input.numel() // N
    i_n = tl.program_id(0)
    i_m = tl.program_id(1)
    o_n = index[i_m]
    o_k = tl.arange(0, K)
    p_in = tl.arange(0, N) + o_n * K
    mask = (o_n < M) & (o_k < K)
    a = tl.load(input + p_in, mask=mask)

    # element-wise equality comparison
    if isinstance(other, torch.Tensor):
        b = tl.load(other + o_k, mask=mask)
    else:
        b = other
    c = a == b

    # write output
    if out is not None:
        p_out = tl.arange(0, M) + i_m * M
        tl.store(out + p_out, c, mask=(i_m < M))

def fused_index_select_eq_wrapper(input, dim, index, other, *, out=None):
    input = input.contiguous()
    index = index.contiguous()
    if isinstance(other, torch.Tensor):
        other = other.contiguous()
    M = index.shape[0]
    K = input.numel() // input.shape[dim]
    if out is None:
        out = torch.empty((M,), dtype=torch.bool, device=input.device)
    fused_index_select_eq[(M, K)](input, dim, index, other, out=out)
    return out
