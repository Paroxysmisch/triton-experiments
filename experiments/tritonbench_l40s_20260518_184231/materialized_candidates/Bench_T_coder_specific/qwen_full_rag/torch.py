import torch
import triton
import triton.language as tl


@triton.jit
def permute_copy_kernel(src, dst, N, K: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset_n = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset_n < N
    offset_k = tl.arange(0, K)
    src_offset = offset_n[:, None] * K + offset_k[None, :]
    dst_offset = offset_k[:, None] * N + offset_n[None, :]
    src_ptrs = src + src_offset
    dst_ptrs = dst + dst_offset
    a = tl.load(src_ptrs, mask=mask, other=0)
    tl.store(dst_ptrs, a, mask=mask)


def torch_permute_copy(inp, dims):
    assert inp.is_contiguous()
    assert min(dims) >= 0
    assert max(dims) < inp.ndim
    assert len(set(dims)) == len(dims)

    shape = list(inp.shape)
    dims_order = sorted(range(len(shape)), key=lambda k: dims[k])
    n = len(shape)
    out_shape = [shape[i] for i in dims]
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    K = 1
    for i in range(n - 1, -1, -1):
        if i not in dims_order:
            dst_idx = dims.index(i)
            out_stride = out.stride()[dst_idx]
            inp_stride = inp.stride()[i]
            assert out_stride == inp_stride
            out = out.contiguous()
            out = out.squeeze(dst_idx)
            out = out.unsqueeze(i)
        else:
            K *= shape[i]

    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), )
    permute_copy_kernel[grid](inp, out, n, K, BLOCK_SIZE=n)
    return out
