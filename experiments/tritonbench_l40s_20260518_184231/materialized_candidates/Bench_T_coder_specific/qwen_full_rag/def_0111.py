import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['A']
)
@triton.jit
def svd_acc_(U, S, VT, A, stride_am, stride_ak, stride_un, stride_uk, stride_vk, stride_vn, M, N, K, k, full_matrices, pid_m, pid_n, **meta):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    rows_per_program = (M + pid_m * N) // N - ((pid_m + 1) * N - 1)
    cols_per_program = (N + pid_n * K) // K - ((pid_n + 1) * K - 1)
    offs_am = (pid_m * rows_per_program + tl.arange(0, rows_per_program)) % M
    offs_bn = (pid_n * cols_per_program + tl.arange(0, cols_per_program)) % N
    offs_k = tl.arange(0, meta['BLOCK_SIZE_K'])
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    u_ptrs = U + (offs_am[:, None] * stride_un + offs_k[None, :] * stride_uk)
    v_ptrs = VT + (offs_k[:, None] * stride_vk + offs_bn[None, :] * stride_vn)
    ones = tl.ones((rows_per_program, cols_per_program), dtype=tl.float32)
    zeros = tl.zeros((rows_per_program, cols_per_program), dtype=tl.float32)
    mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    acc = tl.load(acc)
    au = tl.load(a_ptrs, mask=mask, other=zeros.to(A.dtype.element_ty))
    av = tl.load(v_ptrs, mask=mask, other=zeros.to(VT.dtype.element_ty))
    acc += tl.dot(au, av, allow_tf32=False)
    if full_matrices:
        tl.store(u_ptrs, acc.to(U.dtype.element_ty), mask=(offs_am[:, None] < M) & (offs_k[None, :] < k))
    else:
        tl.store(u_ptrs, acc.to(U.dtype.element_ty), mask=(offs_k[:, None] < k) & (offs_k[None, :] < k))


def low_rank_svd_approximation(A: torch.Tensor, k: int, *, full_matrices: bool = True, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    device = A.device
    dtype = A.dtype
    if not dtype.is_fp64():
        if dtype.is_bf16():
            dtype = torch.float32
        A = A.to(dtype)
    if A.ndim < 2:
        raise ValueError("Singular value decomposition is only defined for 2D tensors or batches of 2D tensors")
    batch_shape = A.shape[:-2]
    A = A.reshape(-1, A.shape[-2], A.shape[-1])
    M, N = A.shape[-2:]
    if k < 1 or k > min(M, N):
        raise ValueError(f"Invalid number of singular values {k}: must be between 1 and min({M}, {N})")
    pinv_A_t = torch.linalg.pinv(A)
    if out is None:
        q = torch.empty((M, k), device=device, dtype=dtype)
    else:
        if out.shape != (M, k):
            raise ValueError(f"Expected output tensor of shape ({M}, {k}), but got {out.shape}")
        if out.device != device or out.dtype != dtype:
            raise ValueError(f"Output tensor must reside on {device} and have {dtype} type")
        q = out
    if full_matrices:
        matmul_fn = partial(torch.nn.functional.linear, weight=q)
        trans_matmul_fn = matmul_fn
    else:
        matmul_fn = partial(torch.matmul, other=q)
        trans_matmul_fn = partial(torch.matmul, transpose_b=True, other=q)
    update_output = True
    for n in range(0, N, k):
        block_range = slice(n, min(N, n + k))
        A_block = A[..., block_range]
        u, s, vh = torch.svd(A_block, full_matrices=full_matrices)
        s_inv = torch.reciprocal(s)
        if update_output:
            q.copy_(vh.T)
            B = A_block @ q
            update_output = False
        else:
            svd_acc_(q, s_inv, vh, A_block, A_block.stride(-2), A_block.stride(-1), q.stride(-2), q.stride(-1), vh.stride(-2), vh.stride(-1), M, N, K, k, full_matrices)
            B = matmul_fn(A_block)
        if n == 0:
            C = B
        else:
            C = torch.cat((C, B), dim=-1)
    q = q.to(dtype)
    r = pinv_A_t @ C
    r *= (N / torch.arange(1, k + 1))
    return r.reshape([*batch_shape, *r.shape[-2:]])
