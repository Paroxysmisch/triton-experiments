import torch
import triton
import triton.language as tl

@triton.jit
def splat(
    f32_val,
    f32_out,
    stride_row,
    stride_col,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    x = tl.full((2, BLOCK_SIZE_COL), f32_val, dtype=tl.float32)
    offs_row = 2 * pid0 + tl.arange(0, 2)
    offs_col = tl.arange(0, BLOCK_SIZE_COL)
    a_ptrs = f32_out + (offs_row[:, None] * stride_row + offs_col[None, :] * stride_col)
    tl.store(a_ptrs, x)

def index_fill_(self, dim, index, value):
    # Check if the dim is valid
    if dim < 0 or dim >= self.ndim:
        raise IndexError("dim out of range")

    # Get the size of the dimension
    size = self.size(dim)

    # Create a block of memory to fill
    block = torch.full((size,), value, device=self.device, dtype=self.dtype)

    # Get the strides for the dimension
    stride = self.stride(dim)

    # Get the block size
    block_size = size

    # Launch the splat kernel
    splat[block_size](
        value,
        self.data_ptr() + dim * stride,
        stride,
        1,
        BLOCK_SIZE_ROW=block_size,
        BLOCK_SIZE_COL=1,
    )

    # Fill in the indices
    for i in index:
        self.data_ptr()[i * stride + dim] = value

    return self
