import torch
import triton
import triton.language as tl

@triton.jit
def index_fill_kernel(
    input_ptr,
    index_ptr,
    value,
    n_indices,
    dim,
    input_ndim,
    strides_ptr,
    shapes_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_other = 1
    for i in range(input_ndim):
        if i != dim:
            num_other *= tl.load(shapes_ptr + i)
    total_elements = n_indices * num_other
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    idx_flat = tl.where(mask, offsets, 0)
    idx_index = idx_flat % n_indices
    idx_other = idx_flat // n_indices

    index_val = tl.load(index_ptr + idx_index, mask=mask, other=0)
    offset = 0
    current_idx = idx_other
    for d in range(input_ndim):
        if d == dim:
            stride = tl.load(strides_ptr + d)
            offset += index_val * stride
        else:
            size = tl.load(shapes_ptr + d)
            stride = tl.load(strides_ptr + d)
            coord = current_idx % size
            offset += coord * stride
            current_idx = current_idx // size
    tl.store(input_ptr + offset, value, mask=mask)

def index_fill_(self, dim, index, value):
    assert index.dim() == 1, "index must be 1D"
    assert dim >= 0 and dim < self.dim(), "dim out of range"
    index = index.to(device=self.device)
    n_indices = index.numel()
    if n_indices == 0:
        return self
    input_ndim = self.dim()
    shapes = torch.tensor(self.shape, device=self.device, dtype=torch.int64)
    strides = torch.tensor(self.stride(), device=self.device, dtype=torch.int64)
    num_other = 1
    for d in range(self.dim()):
        if d != dim:
            num_other *= self.size(d)
    total_elements = n_indices * num_other
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    index_fill_kernel[grid](
        self.data_ptr(),
        index.data_ptr(),
        value,
        n_indices,
        dim,
        input_ndim,
        strides.data_ptr(),
        shapes.data_ptr(),
        BLOCK_SIZE=1024,
    )
    return self

# Attach the function to the Tensor class
torch.Tensor.index_fill_ = index_fill_
