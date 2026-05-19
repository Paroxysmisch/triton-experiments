import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_DMODEL': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_DMODEL': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'D'],
)
@triton.jit
def _score_kernel(
    Q, K, M, Out,
    stride_qm, stride_qd,  # Strides for Q
    stride_kn, stride_kd,  # Strides for K
    stride_mn,            # Stride for mask
    stride_om, stride_on, # Strides for output
    M, N, D,             # Dimensions
    scale,               # Scaling factor
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid // num_pid_m
    pid_n = pid % num_pid_m

    # Block pointers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Load mask
    mask = tl.load(M + offs_m[:, None] * stride_mn + offs_n[None, :])
    
    # Iterate over D dimension
    for d in range(0, D, BLOCK_DMODEL):
        # Load Q and K blocks
        q = tl.load(Q + offs_m[:, None] * stride_qm + (d + offs_d)[None, :] * stride_qd,
                   mask=(offs_m[:, None] < M) & ((d + offs_d)[None, :] < D))
        k = tl.load(K + (d + offs_d)[:, None] * stride_kd + offs_n[None, :] * stride_kn,
                   mask=((d + offs_d)[:, None] < D) & (offs_n[None, :] < N))
        
        # Compute block dot product
        acc += tl.dot(q, k)
    
    # Apply scale and mask
    acc = acc * scale
    acc = acc + mask
    
    # Store result
    out_ptr = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptr, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def get_score(q, k, mask, scale=1.0):
    # Extract dimensions
    M, D = q.shape
    N, D_ = k.shape
    assert D == D_, f"Incompatible dimensions: Q:{q.shape} K:{k.shape}"
    
    # Allocate output
    output = torch.empty((M, N), device=q.device, dtype=q.dtype)
    
    # Handle non-contiguous inputs
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not mask.is_contiguous():
        mask = mask.contiguous()
    
    # Configure grid
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )
    
    # Launch kernel
    _score_kernel[grid](
        q, k, mask, output,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        mask.stride(0),
        output.stride(0), output.stride(1),
        M, N, D,
        scale,
    )
    
    return output
