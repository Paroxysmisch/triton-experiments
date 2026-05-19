import torch
import triton
import triton.language as tl
from triton import Config
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

def get_configs_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_SIZE_M': 8},
                               num_stages=num_stages, num_warps=num_warps)
                    )
    return configs

@triton.autotune(
    configs=get_configs_io_bound(),
    key=['M', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': estimate_matmul_time,
        'top_k': 10,
    },
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % args['BLOCK_K'] == 0,
})
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    acc_dtype: tl.constexpr,
    AB_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m_t = pid // num_pid_n
    pid_n_t = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m_t, pid_n_t, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

    for k in range(0, K, BLOCK_K):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = K - k
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        if AB_DTYPE is not None:
            a = a.to(AB_DTYPE)
            b = b.to(AB_DTYPE)
        acc += tl.dot(a, b, out_dtype=acc_dtype)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = acc.to(tl.typeof(c_ptr).element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)

def triton_matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    _, N = b.shape

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    output_dtype = a.dtype
    ab_dtype = output_dtype
    if a.dtype != b.dtype:
        output_dtype = torch.promote_types(a.dtype, b.dtype)
        ab_dtype = output_dtype

    supported_acc_dtypes = {
        torch.float16: tl.float32,
        torch.bfloat16: tl.float32,
        torch.float32: tl.float32,
        torch.int8: tl.int32,
    }
    acc_dtype = supported_acc_dtypes.get(a.dtype, tl.float32)

    c = torch.empty((M, N), device=a.device, dtype=output_dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']), )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        acc_dtype=acc_dtype,
        AB_DTYPE=ab_dtype if ab_dtype != output_dtype else None,
        GROUP_SIZE_M=8,
    )
    return c

# Example test
def test():
    a = torch.randn(512, 256, device='cuda', dtype=torch.float16)
    b = torch.randn(256, 384, device='cuda', dtype=torch.float16)
    c_triton = triton_matmul(a, b)
    c_torch = torch.matmul(a, b)
    print(f"Triton output shape: {c_triton.shape}")
    print(f"Torch output shape: {c_torch.shape}")
    assert torch.allclose(c_triton, c_torch, atol=1e-2, rtol=1e-2), "Outputs do not match"
    print("✅ Triton and Torch outputs matched")

test()
