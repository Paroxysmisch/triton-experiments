import torch
import triton
import triton.language as tl

@triton.jit
def index_select_cat_bwd_kernel(
    grad_source_ptr,    # Pointer to gradient source tensor [M, D]
    index_ptr,         # Pointer to index tensor [N]
    grad_output_ptr,   # Pointer to gradient output tensor [N, D]
    M,                 # First dimension of grad_source
    N,                 # Length of index tensor
    D,                 # Feature dimension
    stride_gs0, stride_gs1,  # Strides of grad_source
    stride_idx,        # Stride of index tensor
    stride_go0, stride_go1,  # Strides of grad_output
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelization
):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate number of elements per block
    num_elements = N * D
    
    # Calculate start offset for this program instance
    offset = pid * BLOCK_SIZE
    
    # Generate offsets for this block
    offs = offset + tl.arange(0, BLOCK_SIZE)
    
    # Mask for bounds checking
    mask = offs < num_elements
    
    # Calculate index and feature positions
    idx_n = offs // D
    idx_d = offs % D
    
    # Load indices for this block
    indices = tl.load(index_ptr + idx_n * stride_idx, mask=mask & (idx_d == 0))
    
    # Load grad_output values
    grad_out = tl.load(
        grad_output_ptr + idx_n * stride_go0 + idx_d * stride_go1,
        mask=mask
    )
    
    # Calculate output positions
    out_pos = indices * stride_gs0 + idx_d * stride_gs1
    
    # Atomic add to grad_source
    tl.atomic_add(
        grad_source_ptr + out_pos,
        grad_out,
        mask=mask
    )

def index_select_cat_bwd(grad_source: torch.Tensor,
                        index: torch.Tensor,
                        grad_output: torch.Tensor):
    """
    Backward pass for concatenated index select operation.
    
    Args:
        grad_source: Gradient tensor of shape [M, D]
        index: Index tensor of shape [N]
        grad_output: Gradient output tensor of shape [N, D]
    """
    assert grad_source.is_cuda and index.is_cuda and grad_output.is_cuda
    assert grad_source.dim() == 2 and index.dim() == 1 and grad_output.dim() == 2
    assert index.shape[0] == grad_output.shape[0]
    assert grad_source.shape[1] == grad_output.shape[1]
    
    M, D = grad_source.shape
    N = index.shape[0]
    
    # Ensure contiguous tensors for efficient memory access
    grad_source = grad_source.contiguous()
    index = index.contiguous()
    grad_output = grad_output.contiguous()
    
    # Calculate optimal block size (power of 2)
    BLOCK_SIZE = triton.next_power_of_2(min(N * D, 1024))
    
    # Calculate grid size
    grid = lambda meta: (triton.cdiv(N * D, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    index_select_cat_bwd_kernel[grid](
        grad_source.data_ptr(),
        index.data_ptr(),
        grad_output.data_ptr(),
        M,
        N,
        D,
        grad_source.stride(0),
        grad_source.stride(1),
        index.stride(0),
        grad_output.stride(0),
        grad_output.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return grad_source
