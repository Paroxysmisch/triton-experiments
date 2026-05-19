import torch
import triton
import triton.language as tl

@triton.jit
def index_select_cat_bwd_kernel(
    index,
    grad_output,
    grad_source,
    stride_index_b, stride_index_f,
    stride_grad_output_b, stride_grad_output_f,
    stride_grad_source_b, stride_grad_source_f,
    BLOCK_SIZE_INDEX: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    pid_batch = tl.program_id(axis=1)
    pid_feat = tl.program_id(axis=2)
    # compute row offsets of the index and grad_output
    offset_index = pid_batch * stride_index_b + tl.arange(0, BLOCK_SIZE_INDEX) * stride_index_f
    offset_grad_output = pid_batch * stride_grad_output_b + tl.arange(0, BLOCK_SIZE_INDEX) * stride_grad_output_f
    # compute col offsets of the index and grad_output
    offset_index_col = tl.arange(0, BLOCK_SIZE_COL)
    offset_grad_output_col = tl.arange(0, BLOCK_SIZE_COL)
    # build a mask of index
    mask_index = offset_index_col[:, None] < BLOCK_SIZE_INDEX
    mask_grad_output = offset_grad_output_col[:, None] < BLOCK_SIZE_INDEX
    # get the index and grad_output
    index = tl.load(index + offset_index[None, :] * stride_index_f + offset_index_col[None, :] * 1, mask=mask_index, other=0)
    grad_output = tl.load(grad_output + offset_grad_output[None, :] * stride_grad_output_f + offset_grad_output_col[None, :] * 1, mask=mask_grad_output, other=0)
    # compute col offsets of the grad_source
    offset_grad_source = index * stride_grad_source_f
    offset_grad_source_col = offset_grad_output_col
    # build a mask of grad_source
    mask_grad_source = offset_grad_source_col[None, :] < BLOCK_SIZE_COL
    # atomic add the grad_output to grad_source
    tl.atomic_add(grad_source + offset_grad_source[:, None] * 1 + offset_grad_source_col[None, :] * stride_grad_source_f, grad_output, mask=mask_grad_source)

def index_select_cat_bwd(index, grad_output):
    # check index and grad_output
    if index.ndim != 2:
        raise ValueError("index tensor should be 2D")
    if grad_output.ndim != 2:
        raise ValueError("grad_output tensor should be 2D")
    if index.stride(0) != 1 or index.stride(1) != grad_output.stride(0):
        index = index.contiguous()
    if grad_output.stride(0) != 1 or grad_output.stride(1) != index.stride(1):
        grad_output = grad_output.contiguous()
    if not index.is_cuda or not grad_output.is_cuda:
        raise ValueError("index and grad_output should be CUDA tensors")
    if index.size(0) != grad_output.size(0) or index.size(1) != grad_output.size(1):
        raise ValueError("index and grad_output should have the same size")
    # compute the shape of grad_source
    grad_source_shape = list(index.shape)
    grad_source_shape[1] = grad_output.size(1)
    # create grad_source
    grad_source = torch.empty(grad_source_shape, dtype=grad_output.dtype, device=grad_output.device)
    # check if we can use empty grid
    if index.size(1) == 0:
        return grad_source
    # define the grid
    grid = lambda meta: (index.size(0), triton.cdiv(index.size(1), meta["BLOCK_SIZE_INDEX"]), 1)
    # define the kernel
    index_select_cat_bwd_kernel[grid](
        index,
        grad_output,
        grad_source,
        index.stride(0), index.stride(1),
        grad_output.stride(0), grad_output.stride(1),
        grad_source.stride(0), grad_source.stride(1),
    )
    return grad_source
