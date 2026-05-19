import triton
import triton.language as tl
import torch

def index_select_cat_bwd(grad_output, index, grad_source):
    # Check input shapes and types
    assert grad_output.dim() == 2, "grad_output must be a 2D tensor"
    assert index.dim() == 1, "index must be a 1D tensor"
    assert grad_source.dim() == 2, "grad_source must be a 2D tensor"
    assert grad_output.device.type == 'cuda', "grad_output must be on CUDA"
    assert index.device.type == 'cuda', "index must be on CUDA"
    assert grad_source.device.type == 'cuda', "grad_source must be on CUDA"
    assert grad_output.stride(0) == grad_source.stride(0), "grad_output and grad_source must have the same stride along the first dimension"
    assert grad_output.stride(1) == grad_source.stride(1), "grad_output and grad_source must have the same stride along the second dimension"

    # Get tensor dimensions
    N, K = grad_output.shape
    M, K = grad_source.shape

    # Define the Triton kernel
    index_select_cat_bwd_kernel = triton.compile(index_select_cat_bwd_kernel, 
                                               num_warps=4, 
                                               num_stages=2)

    # Launch the kernel
    grid_size = (N + 31) // 32
    index_select_cat_bwd_kernel[grid_size, BLOCK_SIZE_INDEX, BLOCK_SIZE_COL](grad_output, index, grad_source, N, M, K)

    return grad_source
