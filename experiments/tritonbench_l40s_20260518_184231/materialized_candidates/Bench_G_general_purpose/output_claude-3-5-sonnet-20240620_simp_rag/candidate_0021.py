import triton
import triton.language as tl
import torch

@triton.jit
def bmm_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, o_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides for the batch dimension
    stride_ab, stride_bb, stride_ob,
    # Matrix strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    """Kernel for computing the batched matrix multiplication C = A @ B.
    A has shape (batch, M, K), B has shape (batch, K, N)
    """
    # -----------------------------------------------------------
    # Matrix multiplication computation
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block pointers
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate to compute a block of the C matrix
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_idx = k * BLOCK_K
        # Load A and B blocks
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + k_idx * stride_ak,
                   mask=offs_am[:, None] < M, other=0.0)
        b = tl.load(b_ptr + k_idx * stride_bk + offs_bn[None, :] * stride_bn,
                   mask=offs_bn[None, :] < N, other=0.0)
        # Compute block multiplication
        acc += tl.dot(a, b)
    
    # Write result
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(o_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc, mask=mask)

def bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched matrix multiplication using Triton kernel."""
    # Check input dimensions
    assert a.dim() == 3 and b.dim() == 3, "Input tensors must be 3D"
    assert a.shape[0] == b.shape[0], "Batch sizes must match"
    assert a.shape[2] == b.shape[1], "Incompatible matrix dimensions"
    
    # Matrix dimensions
    batch, M, K = a.shape
    _, K, N = b.shape
    
    # Kernel meta-parameters
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16
    GROUP_M = 8
    
    # Allocate output
    o = torch.empty((batch, M, N), device=a.device, dtype=a.dtype)
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    bmm_kernel[grid](
        a_ptr=a, b_ptr=b, o_ptr=o,
        M=M, N=N, K=K,
        stride_ab=a.stride(0), stride_bb=b.stride(0), stride_ob=o.stride(0),
        stride_am=a.stride(1), stride_ak=a.stride(2),
        stride_bk=b.stride(1), stride_bn=b.stride(2),
        stride_om=o.stride(1), stride_on=o.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
    )
    
    return o
