import torch
import triton
import triton.language as tl


@triton.jit
def _index_fill_kernel(
    inp,
    index,
    value,
    stride,
    dim,
    index_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    rows_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = rows_offset < index_size

    input_ptr = inp + rows_offset * stride
    idx_ptr = index + rows_offset
    val = tl.load(value)
    cur_index = tl.load(idx_ptr, mask=mask)

    tl.store(input_ptr + cur_index, val, mask=mask)


def index_fill_(self, dim, index, value):
    assert (
        index.is_contiguous()
    ), "Index tensor must be contiguous. Use index.contiguous() to make it contiguous"
    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    assert index.dtype == torch.int64, "Index tensor must have dtype torch.int64"
    assert (
        index.shape == list(index.strides) + [1]
    ), "The shape of index should be broadcastable with self.shape except in the dimension being indexed"

    value = torch.tensor(value, device=self.device, dtype=self.dtype)
    inp_shape_flat = list(self.shape)
    inp_strides_flat = list(self.stride())
    inp_shape_flat[dim] = 1

    N = 1
    for s in index.size():
        N *= s
    M = triton.cdiv(index.numel(), index.size()[dim])

    BLOCK_SIZE = 128
    num_warps = 8

    _index_fill_kernel[M, ](
        self,
        index,
        value,
        inp_strides_flat[dim],
        dim,
        N,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return self
