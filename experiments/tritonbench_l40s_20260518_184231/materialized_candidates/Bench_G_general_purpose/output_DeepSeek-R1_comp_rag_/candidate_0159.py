import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul4_kernel(
    a_ptr, b_quant_ptr, scales_ptr, z_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_sn, stride_sg,
    stride_zn, stride_zg,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
    group_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_quant_ptrs = b_quant_ptr + offs_bn[None, :] * stride_bn + (offs_k[:, None] // (8 * (group_size // BLOCK_SIZE_K))) * stride_bk
    scale_ptrs = scales_ptr + offs_bn[None, :] * stride_sn + (offs_k[:, None] // group_size) * stride_sg
    z_ptrs = z_ptr + offs_bn[None, :] * stride_zn + (offs_k[:, None] // group_size) * stride_zg

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b_quant = tl.load(b_quant_ptrs, mask=offs_k[:, None] < K - k, other=0)
        scales = tl.load(scale_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        zeros = tl.load(z_ptrs, mask=offs_k[:, None] < K - k, other=0.0)

        b_quant = (b_quant >> (offs_k[:, None] % (group_size // BLOCK_SIZE_K) * 4)) & 0xF
        b_quant = (b_quant.to(tl.float32) * scales + zeros

        accumulator += tl.dot(a, b_quant)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_quant_ptrs += BLOCK_SIZE_K // (group_size // BLOCK_SIZE_K) * stride_bk
        scale_ptrs += BLOCK_SIZE_K // group_size * stride_sg
        z_ptrs += BLOCK_SIZE_K // group_size * stride_zg

    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

def matmul_dequantize_int4_gptq(a, b_quant, scales, zeros, group_size, activation="", output=None):
    M, K = a.shape
    N, _ = b_quant.shape
    if output is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        c = output
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul4_kernel[grid](
        a, b_quant, scales, zeros, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b_quant.stride(0), b_quant.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation,
        group_size=group_size
    )
    return c

def quantize_int4(weight: torch.Tensor, group_size: int = 64):
    K, N = weight.shape
    weight = weight.t().contiguous()
    num_groups = (K + group_size - 1) // group_size
    quantized = []
    scales = []
    zeros = []
    for n in range(N):
        for g in range(num_groups):
            start = g * group_size
            end = min(start + group_size, K)
            group = weight[n, start:end]
            max_val, min_val = group.max(), group.min()
            scale = (max_val - min_val) / 15.0
            zero = min_val
            q_group = torch.clamp(torch.round((group - zero) / scale), -8, 7).to(torch.int8)
            packed = torch.zeros((q_group.numel() + 7) // 8, dtype=torch.int32)
            for i in range(q_group.numel()):
                packed[i // 8] |= (q_group[i].int() & 0xF) << ((i % 8) * 4)
            quantized.append(packed)
            scales.append(scale)
            zeros.append(zero)
    quantized = torch.stack(quantized).view(N, -1)
    scales = torch.tensor(scales, dtype=torch.float16, device=weight.device).view(N, num_groups)
    zeros = torch.tensor(zeros, dtype=torch.float16, device=weight.device).view(N, num_groups)
    return quantized, scales, zeros
