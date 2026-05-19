import torch
import triton
import triton.language as tl
from .utils import get_kernel_meta
from ..constants import MAX_N
from ..stream import _current_stream as current_stream

# Triton kernel for computing the Singular Value Decomposition (SVD)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=2),
    ],
    key=["M", "N", "K"],
    prune_configs_by={
        "early_config_prune": lambda configs, named_args: [
            config
            for config in configs
            if (
                config.kwargs["BLOCK_SIZE_M"] * config.kwargs["BLOCK_SIZE_N"]
                <= named_args["M"] * named_args["N"]
            )
        ],
    },
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_SIZE_K"]) == 0,
    }
)
@triton.jit
def _svd(
    M,
    N,
    K,
    A,
    S,
    U,
    V,
    stride_am,
    stride_ak,
    stride_sn,
    stride_sk,
    stride_un,
    stride_uk,
    stride_vn,
    stride_vk,
    dot_dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ASPECT_RATIO: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (
        offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    )  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    v_ptrs = V + (
        offs_bn[:, None] * stride_vn + offs_k[None, :] * stride_vk
    )  # (BLOCK_SIZE_N, BLOCK_SIZE_K)
    s_ptrs = S + offs_bn * stride_sn  # (BLOCK_SIZE_N,)
    u_offs = (
        offs_am * stride_un
    )  # (BLOCK_SIZE_M,) u is outputed as a flat array, so we don't need stride_uk here

    shifter = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(v_ptrs)  # (BLOCK_SIZE_N, BLOCK_SIZE_K)
        acc_dtype = a.dtype.element_ty
        if acc_dtype in [tl.bfloat16, tl.float32]:
            a_fp32 = a.to(tl.float32)
            b_fp32 = b.to(tl.float32)
            accum += tl.dot(a_fp32, b_fp32, allow_tf32=True)
        else:
            accum += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        v_ptrs += BLOCK_SIZE_K * stride_vk

    orig_accum = accum
    if ASPECT_RATIO >= 2:
        dim = MIN_DIM * 2
        chunk_size = MAX_N // dim
        while chunk_size > 0:
            temp_storage = tl.full([BLOCK_SIZE_M, chunk_size], 0.0, dtype=tl.float32)
            intermediate_storage = tl.full(
                [BLOCK_SIZE_M, chunk_size], 0.0, dtype=orig_accum.dtype
            )
            for off in range(0, dim, chunk_size):
                chunk = orig_accum[:, off : off + chunk_size]
                mask = off + chunk_size > dim
                temp_storage = tl.where(mask, temp_storage, chunk.to(temp_storage.dtype))
                intermediate_storage = tl.where(
                    mask,
                    intermediate_storage,
                    tl.dot(orig_accum, chunk.to(intermediate_storage.dtype), allow_tf32=True),
                )
            orig_accum = intermediate_storage * temp_storage
            chunk_size //= 2
    elif ASPECT_RATIO <= 0.5:
        dim = MIN_DIM // 2
        chunk_size = MAX_N // dim
        while chunk_size > 0:
            temp_storage = tl.full([chunk_size, BLOCK_SIZE_N], 0.0, dtype=tl.float32)
            intermediate_storage = tl.full(
                [chunk_size, BLOCK_SIZE_N], 0.0, dtype=orig_accum.dtype
            )
            for off in range(0, dim, chunk_size):
                chunk = orig_accum[off : off + chunk_size, :]
                mask = off + chunk_size > dim
                temp_storage = tl.where(mask, temp_storage, chunk.to(temp_storage.dtype))
                intermediate_storage = tl.where(
                    mask,
                    intermediate_storage,
                    tl.dot(chunk.to(intermediate_storage.dtype), orig_accum, allow_tf32=True),
                )
            orig_accum = intermediate_storage * temp_storage
            chunk_size //= 2

    offs_s = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    s = tl.load(s_ptrs + offs_s)

    if pid_m < num_pid_m and pid_n < num_pid_n:
        u = tl.store(U + u_offs + offs_s[None, :], orig_accum.to(U.dtype.element_ty))

    s = s.to(orig_accum.dtype)
    accum = orig_accum * s.to(orig_accum.dtype)

    v_ptrs = V + (
        offs_bn[None, :] * stride_vn + offs_k[:, None] * stride_vk
    )  # (BLOCK_SIZE_N, BLOCK_SIZE_K)
    a_ptrs = A + (
        offs_k[:, None] * stride_ak + offs_bn[None, :] * stride_bn
    )  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        b = tl.load(v_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K), other=0.0)
        a = tl.load(a_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K), other=0.0)
        accum += tl.dot(b, a, allow_tf32=True, acc_dtype=a.dtype)
        v_ptrs += BLOCK_SIZE_K * stride_vk
        a_ptrs += BLOCK_SIZE_K * stride_ak


def svd(
    A: torch.Tensor,
    full_matrices: bool = True,
    *,
    driver: str = None,
    out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = A.device
    dtype = A.dtype
    assert dtype in [torch
