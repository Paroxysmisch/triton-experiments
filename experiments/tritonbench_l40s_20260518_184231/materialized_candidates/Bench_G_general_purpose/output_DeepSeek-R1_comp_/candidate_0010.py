import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64}, num_warps=4),
        triton.Config({'BLOCK_N': 256, 'BLOCK_M': 64}, num_warps=4),
        triton.Config({'BLOCK_N': 512, 'BLOCK_M': 128}, num_warps=8),
    ],
    key=['N', 'M'],
)
@triton.jit
def mv_kernel(
    A_ptr, B_ptr, C_ptr,
    N, M,
    stride_am, stride_an,
    stride_bm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offs < N
    
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    for m_offset in range(0, M, BLOCK_M):
        m_offs = m_offset + tl.arange(0, BLOCK_M)
        mask_m = m_offs < M
        
        # Load B chunk
        b_ptrs = B_ptr + m_offs * stride_bm
        b = tl.load(b_ptrs, mask=mask_m, other=0.0)
        
        # Load A chunk for all rows in this block
        a_ptrs = A_ptr + (n_offs[:, None] * stride_an + m_offs[None, :] * stride_am)
        a = tl.load(a_ptrs, mask=mask_n[:, None] & mask_m[None, :], other=0.0)
        
        # Compute partial sum
        acc += tl.sum(a * b, axis=1)
    
    # Write back to C
    c_ptrs = C_ptr + n_offs * stride_cn
    tl.store(c_ptrs, acc, mask=mask_n)

def mv(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
    assert A.is_cuda and B.is_cuda and C.is_cuda, "Inputs must be on GPU"
    assert A.dim() == 2, "Matrix A must be 2D"
    assert B.dim() == 1, "Vector B must be 1D"
    assert C.dim() == 1, "Vector C must be 1D"
    assert A.shape[1] == B.shape[0], "Matrix-vector dimensions mismatch"
    assert A.shape[0] == C.shape[0], "Output dimension mismatch"
    
    N, M = A.shape
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']),)
    mv_kernel[grid](A, B, C, N, M,
                    A.stride(1), A.stride(0),
                    B.stride(0),
                    C.stride(0))
    return C
