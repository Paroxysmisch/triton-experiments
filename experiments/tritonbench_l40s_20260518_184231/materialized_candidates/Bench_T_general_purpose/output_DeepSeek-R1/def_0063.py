import torch
import triton
import triton.language as tl
from typing import Union, Tuple, List
from torch import Tensor

@triton.jit
def tensordot_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0)
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0)
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    c = accumulator.to(a.dtype.element_ty)
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))

def triton_matmul(a: Tensor, b: Tensor) -> Tensor:
    assert a.dim() == 2 and b.dim() == 2, "Inputs must be 2D tensors"
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, f"Incompatible dimensions for matmul: {K} vs {K_}"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    tensordot_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32
    )
    return c

def parse_dims(a: Tensor, b: Tensor, dims) -> Tuple[List[int], List[int]]:
    if isinstance(dims, int):
        a_dims = list(range(a.dim() - dims, a.dim()))
        b_dims = list(range(dims))
    elif isinstance(dims, (tuple, list)) and len(dims) == 2:
        a_dims, b_dims = dims[0], dims[1]
    else:
        raise ValueError("dims must be an int or a tuple/list of two lists")
    a_dims = sorted(a_dims)
    b_dims = sorted(b_dims)
    return a_dims, b_dims

def tensordot(a: Tensor, b: Tensor, dims: Union[int, Tuple[List[int], List[int]], List[List[int]]]) -> Tensor:
    a_dims, b_dims = parse_dims(a, b, dims)
    
    contract_a = [a.shape[d] for d in a_dims]
    contract_b = [b.shape[d] for d in b_dims]
    if contract_a != contract_b:
        raise ValueError(f"Contracted dimensions must match, got {contract_a} and {contract_b}")
    
    non_contract_a = [d for d in range(a.dim()) if d not in a_dims]
    non_contract_b = [d for d in range(b.dim()) if d not in b_dims]
    
    a_perm = a.permute(non_contract_a + a_dims)
    a_shape = (-1, int(torch.prod(torch.tensor(contract_a, device='cpu'))))
    a_2d = a_perm.reshape(a_shape)
    
    b_perm = b.permute(b_dims + non_contract_b)
    b_shape = (int(torch.prod(torch.tensor(contract_b, device='cpu'))), -1)
    b_2d = b_perm.reshape(b_shape)
    
    c_2d = triton_matmul(a_2d, b_2d)
    
    out_shape = [a.shape[d] for d in non_contract_a] + [b.shape[d] for d in non_contract_b]
    return c_2d.reshape(out_shape)
