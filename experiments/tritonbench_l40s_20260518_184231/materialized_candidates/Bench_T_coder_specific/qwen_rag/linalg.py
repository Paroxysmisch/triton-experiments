import torch
import triton
import triton.language as tl

@triton.jit
def det_kernel(output_ptr, input_ptr, n, m, stride_a, stride_out, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, n * m)
    
    row = block_start // m
    col = block_start % m
    
    accum = tl.zeros((1,), dtype=input_ptr.dtype)
    for i in range(col, block_end, m):
        accum += input_ptr[row * stride_a + i]
    
    if block_start == block_end:
        accum = 0
    
    tl.store(output_ptr + pid, accum)

def linalg_det(A, out=None):
    assert len(A.shape) >= 2 and A.shape[-2] == A.shape[-1], "Input must be a square matrix"
    n, m = A.shape[-2:]
    stride_a = A.stride(-2)
    stride_out = out.stride(-1) if out is not None else 1
    
    if out is None:
        out = torch.empty_like(A, dtype=torch.float32)
    
    BLOCK_SIZE = 32
    grid_size = (n * m + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    det_kernel[grid_size](out.data_ptr(), A.data_ptr(), n, m, stride_a, stride_out, BLOCK_SIZE=BLOCK_SIZE)
    
    return out
