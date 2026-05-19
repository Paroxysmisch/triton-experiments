import torch
import triton
import triton.language as tl

@triton.jit
def index_fill_kernel(
    index_size,
    value,
    output_ptr,
    output_row_stride,
    index_ptr,
    index_row_stride,
    dim,
    fill_block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    output_row_offsets = pid * fill_block_size + tl.arange(0, fill_block_size)
    output_mask = output_row_offsets < index_size
    index_row_offsets = tl.arange(0, fill_block_size)
    index_mask = index_row_offsets < fill_block_size
    output_ptrs = output_ptr + output_row_stride * output_row_offsets
    index_ptrs = index_ptr + index_row_stride * index_row_offsets

    if dim == 0:
        tl.store(output_ptrs, value, mask=output_mask)
    else:
        tl.store(output_ptrs, value, mask=index_mask)

def index_fill_(self, dim, index, value):
    assert index.dtype == torch.int64, "index should be a LongTensor"
    assert dim >= -self.ndim and dim < self.ndim, "dim out of range (expected to be in range of [{}, {}], but got {})".format(-self.ndim, self.ndim - 1, dim)
    assert index.numel() > 0, "index should not be an empty tensor"
    index = index.contiguous()
    if index.ndim == 1:
        index_size = index.numel()
    else:
        assert index.size(1) == 1, "index should be 1-dimensional or 2-dimensional with 1 as the second dimension"
        index = index.squeeze(1)
        index_size = index.numel()
    assert index_size <= self.size(dim), "indexing error: the size of index should be less than or equal to size of the given dimension"
    fill_block_size = 32
    grid = lambda meta: (triton.cdiv(index_size, meta["fill_block_size"]),)
    index_fill_kernel[grid](
        index_size,
        value,
        self.data_ptr(),
        self.stride(0),
        index.data_ptr(),
        index.stride(0),
        dim,
        fill_block_size=fill_block_size,
    )
    return self

index_fill_func = IndexFill.apply

def index_fill(self, dim, index, value):
    assert (
        self.is_contiguous()
    ), "Only contiguous tensors can be indexed into by index."
    assert (
        index.is_contiguous()
    ), "Only contiguous tensors can be indexed into by index."
    return IndexFill.apply(self, dim, index, value)
