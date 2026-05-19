import triton
import triton.language as tl
import torch

@triton.jit
def _int8_matmul_rowwise_dequantize_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Scaling factors and bias
    state_x_ptr, state_w_ptr, bias_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides for A, B, C
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_split_k = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_split_k * BLOCK_K + tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        k_remaining = K - k * BLOCK_K * SPLIT_K
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < k_remaining), other=0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining), other=0)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        a_ptrs += BLOCK_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_K * SPLIT_K * stride_bk

    acc = acc.to(tl.float32)
    x_scale = tl.load(state_x_ptr + offs_m)
    w_scale = tl.load(state_w_ptr + offs_n)
    acc *= x_scale[:, None] * w_scale[None, :]

    if ADD_BIAS:
        bias = tl.load(bias_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
                       mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0)
        acc += bias

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc, mask=mask)
    else:
        tl.atomic_add(c_ptrs, acc, mask=mask)

def int8_matmul_rowwise_dequantize(
    a: torch.Tensor,
    b: torch.Tensor,
    state_x: torch.Tensor,
    state_w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 64,
    SPLIT_K: int = 1,
    GROUP_M: int = 8,
    allow_tf32: bool = True,
):
    assert a.is_cuda and b.is_cuda and state_x.is_cuda and state_w.is_cuda
    assert a.dtype == torch.int8 and b.dtype == torch.int8
    assert state_x.dtype == torch.float32 and state_w.dtype == torch.float32
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, "Incompatible dimensions"
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
    
    _int8_matmul_rowwise_dequantize_kernel[grid](
        a, b, c,
        state_x, state_w,
        bias if bias is not None else None,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K,
        GROUP_M=GROUP_M,
        ADD_BIAS=bias is not None,
        ALLOW_TF32=allow_tf32,
    )
    return c
