import torch
import triton
import triton.language as tl
from typing import Union, Tuple, List, Optional

def tensordot(a: torch.Tensor, b: torch.Tensor, dims: Union[int, Tuple[List[int], List[int]], List[List[int]]]) -> torch.Tensor:
    # Parse dims to get contracted dimensions for a and b
    a_dims: List[int] = []
    b_dims: List[int] = []
    if isinstance(dims, int):
        a_dims = list(range(a.dim() - dims, a.dim()))
        b_dims = list(range(dims))
    elif isinstance(dims, (tuple, list)) and len(dims) == 2:
        a_dims, b_dims = dims[0], dims[1]
    else:
        raise ValueError("dims must be an int or a tuple/list of two lists")
    
    # Verify contracted dimensions
    if len(a_dims) != len(b_dims):
        raise ValueError("Number of contracted dimensions must match")
    for a_dim, b_dim in zip(a_dims, b_dims):
        if a.shape[a_dim] != b.shape[b_dim]:
            raise ValueError(f"Dimension size mismatch: a[{a_dim}]={a.shape[a_dim]} vs b[{b_dim}]={b.shape[b_dim]}")
    
    # Compute non-contracted dimensions
    a_non_contract = [d for d in range(a.dim()) if d not in a_dims]
    b_non_contract = [d for d in range(b.dim()) if d not in b_dims]
    
    # Transpose tensors to group non-contracted and contracted dimensions
    a_shaped = a.permute(a_non_contract + a_dims).contiguous()
    b_shaped = b.permute(b_dims + b_non_contract).contiguous()
    
    # Reshape into 2D matrices
    a_2d = a_shaped.view(-1, torch.tensor(a.shape)[a_dims].prod().item())
    b_2d = b_shaped.view(torch.tensor(b.shape)[b_dims].prod().item(), -1)
    
    # Perform matrix multiplication using Triton kernel
    c_2d = triton_matmul(a_2d, b_2d)
    
    # Reshape result to combined non-contracted dimensions
    c_shape = list(a_shaped.shape[:len(a_non_contract)]) + list(b_shaped.shape[len(b_dims):])
    return c_2d.view(c_shape)

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask)

def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matrix multiplication"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    
    ACC_TYPE = tl.float32 if a.dtype in [torch.float16, torch.bfloat16, torch.float32] else tl.int32
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=64,
        ACC_TYPE=ACC_TYPE,
    )
    return c
