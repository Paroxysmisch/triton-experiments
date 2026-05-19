import torch
import triton
import triton.language as tl

@triton.jit
def index_fill_kernel(
    input_ptr,
    indices_ptr,
    value,
    dim_stride,
    other_strides_ptr,
    n_indices,
    n_other_elems,
    other_ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < (n_indices * n_other_elems)
    
    # Calculate idx_idx and sub_idx
    idx_idx = offs // n_other_elems
    sub_idx = offs % n_other_elems
    
    # Load the index value
    index = tl.load(indices_ptr + idx_idx, mask=mask, other=0)
    
    # Compute offset along the target dimension
    dim_offset = index * dim_stride
    
    # Compute offset from other dimensions
    other_offset = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    strides = tl.load(other_strides_ptr + tl.arange(0, other_ndim), 
                      mask=tl.arange(0, other_ndim) < other_ndim)
    remaining = sub_idx
    for i in range(other_ndim - 1, -1, -1):
        stride = strides[i]
        coord = remaining // stride
        remaining = remaining % stride
        other_offset += coord * stride
    
    # Total offset
    total_offset = dim_offset + other_offset
    # Store the value
    tl.store(input_ptr + total_offset, value * tl.ones((BLOCK_SIZE,), dtype=tl.float32), mask=mask)

def index_fill_(self, dim, index, value):
    assert dim >= 0 and dim < self.dim(), "dim out of range"
    assert index.dim() == 1, "index must be a 1D tensor"
    device = self.device
    dtype = self.dtype
    
    # Convert value to float
    value = float(value)
    
    # Original tensor info
    strides = list(self.stride())
    shape = list(self.size())
    dim_stride = strides[dim]
    
    # Other dimensions info
    other_dims = [i for i in range(self.dim()) if i != dim]
    other_strides = [strides[i] for i in other_dims]
    other_ndim = len(other_dims)
    n_other_elems = 1
    for i in other_dims:
        n_other_elems *= shape[i]
    
    # Prepare tensors for Triton
    indices = index.to(device=device)
    other_strides_tensor = torch.tensor(other_strides, device=device, dtype=torch.int64)
    
    # Kernel configuration
    total_elems = indices.numel() * n_other_elems
    BLOCK_SIZE = 128
    grid = lambda meta: (triton.cdiv(total_elems, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    index_fill_kernel[grid](
        self.data_ptr(),
        indices.data_ptr(),
        value,
        dim_stride,
        other_strides_tensor.data_ptr(),
        indices.numel(),
        n_other_elems,
        other_ndim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return self

# Example test
if __name__ == "__main__":
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float, device='cuda')
    index = torch.tensor([0, 2], device='cuda')
    x.index_fill_(1, index, -1)
    print(x.cpu())
