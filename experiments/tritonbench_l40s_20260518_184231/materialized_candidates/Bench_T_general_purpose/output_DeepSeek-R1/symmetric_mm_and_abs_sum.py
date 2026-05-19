import torch
import triton
import triton.language as tl

@triton.jit
def symmetric_gemm_kernel(
    a_ptr, c_ptr, out_ptr,
    n, m,
    stride_am, stride_ak,
    stride_cm, stride_cn,
    stride_out_m, stride_out_n,
    alpha,
    beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = a_ptr + offs_k[:, None] * stride_ak + offs_n[None, :] * stride_am  # B is A.T
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(m, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < m - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < m - k * BLOCK_SIZE_K, other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_am
    
    acc *= alpha
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < n) & (offs_n[None, :] < n)
    c = tl.load(c_ptrs, mask=c_mask, other=0.0)
    
    acc += beta * c
    
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.store(out_ptrs, acc, mask=c_mask)

@triton.jit
def abs_sum_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    abs_values = tl.abs(values)
    sum_part = tl.sum(abs_values, axis=0)
    tl.atomic_add(output_ptr, sum_part)

def symmetric_mm_and_abs_sum(A: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    assert A.dim() == 2, "A must be a 2D tensor"
    n, m = A.shape
    assert C.shape == (n, n), f"C must be ({n}, {n}), but got {C.shape}"
    
    temp = torch.empty((n, n), device=A.device, dtype=A.dtype)
    
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    grid = (
        triton.cdiv(n, BLOCK_SIZE_M),
        triton.cdiv(n, BLOCK_SIZE_N),
    )
    symmetric_gemm_kernel[grid](
        A, C, temp,
        n, m,
        A.stride(0), A.stride(1),
        C.stride(0), C.stride(1),
        temp.stride(0), temp.stride(1),
        alpha,
        beta,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    sum_abs = torch.zeros(1, device=A.device, dtype=torch.float32)
    n_elements = n * n
    BLOCK_SIZE_SUM = 1024
    grid_sum = (triton.cdiv(n_elements, BLOCK_SIZE_SUM),)
    abs_sum_kernel[grid_sum](
        temp.view(-1), sum_abs,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE_SUM,
    )
    
    return sum_abs
