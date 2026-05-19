import torch
import triton
import triton.language as tl
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@triton.jit
def triton_add_mean_kernel(
    in_ptr0,
    in_ptr1,
    out_ptr,
    n_elements,
    in_stride0,
    in_stride1,
    out_stride0,
    out_stride1,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    temp = tl.load(in_ptr0 + (offsets % in_stride0) + (offsets // in_stride0) * in_stride1, mask, other=0.0)
    temp = temp.to(tl.float32)
    other = tl.load(in_ptr1 + (offsets % in_stride0) + (offsets // in_stride0) * in_stride1, mask, other=0.0)
    other = other.to(tl.float32)
    temp = triton_helpers.promote_to_tensor(temp)
    other = triton_helpers.promote_to_tensor(other)
    temp = triton_helpers.add(temp, triton_helpers.mul(other, 1))
    temp = triton_helpers.mean(temp, (0,))
    temp = temp.to(tl.float32)
    tl.store(out_ptr + (offsets % out_stride0) + (offsets // out_stride0) * out_stride1, temp, mask)

def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None) -> torch.Tensor:
    if dtype is None:
        dtype = input.dtype
    if out is None:
        out = torch.empty_like(input, dtype=dtype)
    else:
        assert out.dtype == dtype
    assert input.is_contiguous()
    assert out.is_contiguous()
    in_view_stride = list(input.stride())
    if dim is None:
        in_view_stride = [-1]
    else:
        if isinstance(dim, tuple):
            dim = list(dim)
        else:
            dim = [dim]
        in_view_shape = list(input.shape)
        for d in dim:
            assert d >= -input.dim() and d < input.dim()
            in_view_shape[d] = 1
        in_view = input.reshape(in_view_shape)
        in_view_stride = in_view.stride()
    n_elements = in_view.numel()
    if n_elements == 0:
        return out
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    triton_add_mean_kernel[grid](input, other, out, n_elements, in_view_stride[0], in_view_stride[1], out.stride(0), out.stride(1))
    if dim is not None or keepdim:
        out = out.reshape(input.shape)
    return out
