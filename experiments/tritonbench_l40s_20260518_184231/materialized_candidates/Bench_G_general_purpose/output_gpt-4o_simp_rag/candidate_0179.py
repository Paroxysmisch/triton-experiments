import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
    ],
    key=["N"]
)
@triton.jit
def chunk_global_cumsum_vector_kernel(
    s_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < N
    s = tl.load(s_ptr + offsets, mask=mask, other=0.0)
    
    # Compute cumulative sum within the block
    cumsum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    cumsum = tl.cumsum(s, axis=0)
    
    # Store the result
    tl.store(out_ptr + offsets, cumsum, mask=mask)

def chunk_global_cumsum_vector(s):
    N = s.shape[0]
    out = torch.empty_like(s)
    
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    chunk_global_cumsum_vector_kernel[grid](
        s, out, N,
        num_warps=4,
        BLOCK_SIZE=128
    )
    
    return out
