import torch
import triton
import triton.language as tl

@triton.jit
def _int8_matmul_rowwise_dequantize(
    # Matrix pointers
    A_ptr, B_ptr, C_ptr,
    # Dequantization parameters
    state_x_ptr, state_w_ptr,
    # Optional bias
    Bias_ptr,
    # Matrix dimensions
    M, N, K,
    # Stride info
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Bias strides
    stride_bm, stride_bn,
    # Kernel parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Get program IDs
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    
    # Calculate number of blocks
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Split K dimension
    k_offset = pid_z * BLOCK_K
    K_remaining = K - k_offset
    K_loaded = tl.minimum(BLOCK_K, K_remaining)

    # Create block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = k_offset + tl.arange(0, BLOCK_K)

    # Initialize pointers
    A_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, k_offset),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0)
    )
    
    B_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(k_offset, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1)
    )

    # Load A and B blocks
    a = tl.load(A_block_ptr, boundary_check=(0,1))
    b = tl.load(B_block_ptr, boundary_check=(0,1))

    # Dequantization scaling
    scale_a = tl.load(state_x_ptr + offs_m, mask=offs_m < M, other=0.0)
    scale_b = tl.load(state_w_ptr + offs_k, mask=offs_k < K, other=0.0)
    
    # Apply scaling
    a_scaled = a.to(tl.float32) * scale_a[:, None]
    b_scaled = b.to(tl.float32) * scale_b[:, None]
    
    # Compute accumulator
    acc = tl.dot(a_scaled, b_scaled, allow_tf32=False)

    # Add bias if present
    if Bias_ptr is not None:
        bias_ptr = Bias_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
        bias = tl.load(bias_ptr, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
        acc += bias

    # Calculate output offsets
    c_offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + c_offs_m[:, None] * stride_cm + c_offs_n[None, :] * stride_cn

    # Write back with atomic add for split-K
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc, mask=(c_offs_m[:, None] < M) & (c_offs_n[None, :] < N))
    else:
        tl.atomic_add(c_ptrs, acc, mask=(c_offs_m[:, None] < M) & (c_offs_n[None, :] < N))

def int8_matmul_rowwise_dequantize(
    a: torch.Tensor,
    b: torch.Tensor,
    state_x: torch.Tensor,
    state_w: torch.Tensor,
    bias: torch.Tensor = None,
    SPLIT_K: int = 1,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 64,
    GROUP_M: int = 8,
):
    # Validate inputs
    assert a.dtype == torch.int8 and b.dtype == torch.int8
    assert a.is_cuda and b.is_cuda
    assert state_x.dtype == torch.float32 and state_w.dtype == torch.float32
    
    M, K = a.shape
    K_check, N = b.shape
    assert K == K_check, "Incompatible dimensions"
    
    # Ensure tensors are contiguous
    if not a.is_contiguous(): a = a.contiguous()
    if not b.is_contiguous(): b = b.contiguous()
    if not state_x.is_contiguous(): state_x = state_x.contiguous()
    if not state_w.is_contiguous(): state_w = state_w.contiguous()
    if bias is not None and not bias.is_contiguous(): bias = bias.contiguous()

    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # Calculate grid size
    grid = lambda opt: (
        triton.cdiv(M, opt['BLOCK_M']) * triton.cdiv(N, opt['BLOCK_N']),
        SPLIT_K,
    )

    # Launch kernel
    _int8_matmul_rowwise_dequantize[grid](
        a, b, c,
        state_x, state_w,
        bias if bias is not None else None,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        bias.stride(0), bias.stride(1) if bias is not None else 0, 0,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K,
        GROUP_M=GROUP_M,
    )
    return c
