import triton
import triton.language as tl

@triton.jit
def index_select_cat_bwd_kernel(
    grad_source_ptr,  # Pointer to the gradient of the source tensor
    index_ptr,       # Pointer to the index tensor
    grad_output_ptr, # Pointer to the gradient of the output tensor
    grad_source_stride0,  # Stride of the source tensor in the 0th dimension
    grad_source_stride1,  # Stride of the source tensor in the 1st dimension
    index_stride0,       # Stride of the index tensor in the 0th dimension
    grad_output_stride0,  # Stride of the output tensor in the 0th dimension
    grad_output_stride1,  # Stride of the output tensor in the 1st dimension
    n,                   # Number of elements in the 0th dimension
    m,                   # Number of elements in the 1st dimension
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    for i in range(m):
        index = tl.load(index_ptr + offsets * index_stride0, mask=mask)
        grad_output = tl.load(grad_output_ptr + offsets * grad_output_stride0 + i * grad_output_stride1, mask=mask)
        tl.atomic_add(grad_source_ptr + index * grad_source_stride0 + i * grad_source_stride1, grad_output, mask=mask)

import torch
import triton
import triton.language as tl

class IndexSelectCatBwdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_source, index, grad_output):
        # Check shape and stride compatibility
        assert grad_source.dim() == 2, "grad_source must be a 2D tensor"
        assert index.dim() == 1, "index must be a 1D tensor"
        assert grad_output.dim() == 2, "grad_output must be a 2D tensor"
        assert grad_source.size(0) == index.size(0), "grad_source and index must have the same size in the 0th dimension"
        assert grad_source.size(1) == grad_output.size(1), "grad_source and grad_output must have the same size in the 1st dimension"

        n, m = grad_source.size()
        BLOCK_SIZE = 128

        # Allocate output tensor
        grad_source = grad_source.clone()

        # Define grid and block dimensions
        grid = (triton.cdiv(n, BLOCK_SIZE),)

        # Launch the kernel
        index_select_cat_bwd_kernel[grid](
            grad_source_ptr=grad_source,
            index_ptr=index,
            grad_output_ptr=grad_output,
            grad_source_stride0=grad_source.stride(0),
            grad_source_stride1=grad_source.stride(1),
            index_stride0=index.stride(0),
            grad_output_stride0=grad_output.stride(0),
            grad_output_stride1=grad_output.stride(1),
            n=n,
            m=m,
            BLOCK_SIZE=BLOCK_SIZE
        )

        return grad_source

def index_select_cat_bwd(grad_source, index, grad_output):
    return IndexSelectCatBwdFunction.apply(grad_source, index, grad_output)
