import torch
import triton
import triton.language as tl

@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    alpha,
    beta,
    n, m, p,
    stride_am, stride_ak,
    stride_bk, stride_bp,
    stride_cn, stride_cp,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    if pid_m >= n or pid_n >= p:
        return
    
    offs_m = pid_m
    offs_n = pid_n
    
    sum_val = 0.0
    for k in range(0, m):
        a_offset = offs_m * stride_am + k * stride_ak
        b_offset = k * stride_bk + offs_n * stride_bp
        a = tl.load(a_ptr + a_offset)
        b = tl.load(b_ptr + b_offset)
        sum_val += a * b
    sum_val *= alpha
    
    c_offset = offs_m * stride_cn + offs_n * stride_cp
    c_old = tl.load(c_ptr + c_offset)
    sum_val += beta * c_old
    tl.store(c_ptr + c_offset, sum_val)

@triton.jit
def dot_kernel(
    row0_ptr, row1_ptr, output_ptr,
    length,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < length
    
    a = tl.load(row0_ptr + offsets, mask=mask)
    b = tl.load(row1_ptr + offsets, mask=mask)
    
    product = a * b
    sum_product = tl.sum(product, axis=0)
    
    tl.atomic_add(output_ptr, sum_product)

def matrix_multiply_and_row_dot(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, C: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda and C.is_cuda, "Inputs must be on GPU"
    assert A.shape[0] == C.shape[0], "A and C must have the same number of rows"
    assert A.shape[1] == B.shape[0], "A and B have incompatible dimensions"
    assert B.shape[1] == C.shape[1], "B and C must have the same number of columns"
    assert C.shape[0] >= 2, "C must have at least two rows"
    
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    
    n, m = A.shape
    _, p = B.shape
    
    grid = (n, p)
    gemm_kernel[grid](
        A, B, C,
        alpha,
        beta,
        n, m, p,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=1, BLOCK_SIZE_N=1
    )
    
    row0 = C[0].contiguous()
    row1 = C[1].contiguous()
    length = row0.shape[0]
    
    output = torch.zeros(1, device='cuda', dtype=row0.dtype)
    BLOCK_SIZE = 256
    num_blocks = (length + BLOCK_SIZE - 1) // BLOCK_SIZE
    dot_kernel[num_blocks](
        row0, row1, output,
        length,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output
