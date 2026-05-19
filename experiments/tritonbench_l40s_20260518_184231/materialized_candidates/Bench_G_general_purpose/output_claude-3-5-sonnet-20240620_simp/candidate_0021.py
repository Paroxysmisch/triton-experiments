import torch
import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, o_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in each dimension
    stride_ab, stride_am, stride_ak,  # Strides for A
    stride_bb, stride_bk, stride_bn,  # Strides for B
    stride_ob, stride_om, stride_on,  # Strides for output
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the batched matrix multiplication C = A x B.
    A has shape (batch, M, K), B has shape (batch, K, N) and C has shape (batch, M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We'll advance this pointer as we move in the K direction
    # and accumulate
    batch_id = tl.program_id(1)
    # Offset the pointers for batch dimension
    a_ptr = a_ptr + batch_id * stride_ab
    b_ptr = b_ptr + batch_id * stride_bb
    o_ptr = o_ptr + batch_id * stride_ob
    
    # Offset the pointers for the M and N dimensions
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptr = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptr = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    # We accumulate into a `acc` variable
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, using mask to handle boundary conditions
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptr, mask=offs_k[None, :] < k_remaining)
        b = tl.load(b_ptr, mask=offs_k[:, None] < k_remaining)
        # We accumulate along the K dimension
        acc += tl.dot(a, b)
        # Advance the ptrs to the next K block
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk
    
    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptr = o_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(c_ptr, acc, mask=mask)

def bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched matrix multiplication using Triton kernel.
    
    Args:
        a: Input tensor of shape (batch, M, K)
        b: Input tensor of shape (batch, K, N)
    
    Returns:
        Output tensor of shape (batch, M, N)
    """
    assert a.ndim == b.ndim == 3, "Input tensors must be 3D"
    assert a.shape[0] == b.shape[0], "Batch sizes must match"
    assert a.shape[2] == b.shape[1], "Incompatible matrix dimensions"
    
    # Get input dimensions
    batch, M, K = a.shape
    _, K, N = b.shape
    
    # Ensure contiguous inputs
    a = a.contiguous()
    b = b.contiguous()
    
    # Allocate output
    o = torch.empty((batch, M, N), device=a.device, dtype=a.dtype)
    
    # Define meta-parameters
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    GROUP_SIZE_M = 8
    
    # Calculate grid size
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), batch)
    
    # Launch kernel
    bmm_kernel[grid](
        a, b, o,
        M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return o
