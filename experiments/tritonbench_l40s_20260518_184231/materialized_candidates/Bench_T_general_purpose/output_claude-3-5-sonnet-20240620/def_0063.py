import triton
import triton.language as tl
import torch
from typing import Union, List, Tuple

@triton.jit
def _tensordot_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, out_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides for each tensor
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    """Triton kernel for tensor contraction."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_n
    group_id = pid // num_pid_in_group
    pid_m = group_id
    pid_n = pid % num_pid_in_group

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        mask_k = offs_k[None, :] < K - k
        a = tl.load(a_ptrs, mask=mask_k, other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
        accumulator += tl.dot(a, b)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        
    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + offs_om[:, None] * stride_om + offs_on[None, :] * stride_on
    
    mask_m = offs_om[:, None] < M
    mask_n = offs_on[None, :] < N
    tl.store(out_ptrs, accumulator, mask=mask_m & mask_n)

def tensordot(a: torch.Tensor, b: torch.Tensor, 
              dims: Union[int, Tuple[List[int], List[int]], List[List[int]]]) -> torch.Tensor:
    """
    Returns a contraction of a and b over multiple dimensions.
    
    Args:
        a (Tensor): Left tensor to contract
        b (Tensor): Right tensor to contract
        dims (int or Tuple[List[int], List[int]] or List[List[int]]): number of dimensions 
            to contract or explicit lists of dimensions for a and b respectively
    
    Returns:
        Tensor: The contracted tensor
    """
    # Convert dims to lists of dimensions to contract
    if isinstance(dims, int):
        dims_a = list(range(-dims, 0))
        dims_b = list(range(dims))
    else:
        dims_a, dims_b = dims[0], dims[1]
    
    # Verify dimension compatibility
    if len(dims_a) != len(dims_b):
        raise ValueError("Length of contracted dimensions must match")
    
    # Check contracted dimensions sizes match
    for idx_a, idx_b in zip(dims_a, dims_b):
        if a.shape[idx_a] != b.shape[idx_b]:
            raise ValueError(f"Contracted dimension sizes do not match: {a.shape[idx_a]} != {b.shape[idx_b]}")
    
    # Prepare tensors by moving contracted dimensions to the end/beginning
    a_free_dims = [i for i in range(a.dim()) if i not in dims_a]
    b_free_dims = [i for i in range(b.dim()) if i not in dims_b]
    
    # Permute and reshape tensors
    a_permuted = a.permute((*a_free_dims, *dims_a))
    b_permuted = b.permute((*dims_b, *b_free_dims))
    
    M = torch.prod(torch.tensor([a.shape[i] for i in a_free_dims])) if a_free_dims else 1
    N = torch.prod(torch.tensor([b.shape[i] for i in b_free_dims])) if b_free_dims else 1
    K = torch.prod(torch.tensor([a.shape[i] for i in dims_a]))
    
    # Reshape to 2D matrices for contraction
    a_reshaped = a_permuted.reshape(M, K)
    b_reshaped = b_permuted.reshape(K, N)
    
    # Output tensor
    output = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Grid and block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    _tensordot_kernel[grid](
        a_reshaped, b_reshaped, output,
        M, N, K,
        a_reshaped.stride(0), a_reshaped.stride(1),
        b_reshaped.stride(0), b_reshaped.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # Reshape output to final shape
    final_shape = tuple(a.shape[i] for i in a_free_dims) + tuple(b.shape[i] for i in b_free_dims)
    return output.reshape(final_shape)
