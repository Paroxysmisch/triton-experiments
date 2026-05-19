import torch
import triton
import triton.language as tl

@triton.jit
def batched_vecmat_kernel(
    A_ptr, B_ptr, C_ptr,
    dim_m, dim_n, dim_k,
    stride_am, stride_ak,
    stride_bm, stride_bn, stride_bk,
    stride_cm, stride_cn,
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_n = tl.cdiv(dim_n, block_n)
    m_index = pid // num_blocks_n
    n_index = pid % num_blocks_n
    
    m_start = m_index * block_m
    n_start = n_index * block_n
    
    m_offs = m_start + tl.arange(0, block_m)
    n_offs = n_start + tl.arange(0, block_n)
    
    vecmat = tl.zeros((block_m, block_n), dtype=tl.float32)
    
    num_k_blocks = tl.cdiv(dim_k, block_k)
    for k_block in range(num_k_blocks):
        k_start = k_block * block_k
        k_offs = k_start + tl.arange(0, block_k)
        
        a_ptrs = A_ptr + (m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak)
        a_mask = (m_offs[:, None] < dim_m) & (k_offs[None, :] < dim_k)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        b_ptrs = B_ptr + (m_offs[:, None, None] * stride_bm + n_offs[None, :, None] * stride_bn + k_offs[None, None, :] * stride_bk)
        b_mask = (m_offs[:, None, None] < dim_m) & (n_offs[None, :, None] < dim_n) & (k_offs[None, None, :] < dim_k)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        product = a[:, None, :] * b
        partial = tl.sum(product, axis=2)
        vecmat += partial
    
    c_ptrs = C_ptr + (m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn)
    c_mask = (m_offs[:, None] < dim_m) & (n_offs[None, :] < dim_n)
    tl.store(c_ptrs, vecmat, mask=c_mask)

def batched_vecmat(A: torch.Tensor, B: torch.Tensor, block_m: int, block_n: int, block_k: int) -> torch.Tensor:
    assert A.dim() == 2, "A must be 2D"
    assert B.dim() == 3, "B must be 3D"
    dim_m, dim_k = A.shape
    dim_m_b, dim_n, dim_k_b = B.shape
    assert dim_m == dim_m_b, "M dimension mismatch between A and B"
    assert dim_k == dim_k_b, "K dimension mismatch between A and B"
    
    assert dim_m % block_m == 0, f"dim_m {dim_m} must be divisible by block_m {block_m}"
    assert dim_n % block_n == 0, f"dim_n {dim_n} must be divisible by block_n {block_n}"
    assert dim_k % block_k == 0, f"dim_k {dim_k} must be divisible by block_k {block_k}"
    
    output = torch.empty((dim_m, dim_n), device=A.device, dtype=A.dtype)
    
    num_blocks_m = dim_m // block_m
    num_blocks_n = dim_n // block_n
    grid = (num_blocks_m * num_blocks_n, )
    
    batched_vecmat_kernel[grid](
        A, B, output,
        dim_m, dim_n, dim_k,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        output.stride(0), output.stride(1),
        block_m=block_m, block_n=block_n, block_k=block_k,
    )
    return output
