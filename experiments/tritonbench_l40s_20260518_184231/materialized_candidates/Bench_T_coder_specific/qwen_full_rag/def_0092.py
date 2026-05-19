import torch
import triton
import triton.language as tl

@triton.jit
def _tensordot_rsqrt_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        BLOCK_SIZE_K: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    a_mask = offs_m[:, None] < M and offs_k[None, :] < K
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)

    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
    b_mask = offs_k[:, None] < K and offs_n[None, :] < N
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

    c = tl.dot(a, b)
    c = 1 / tl.sqrt(c)
    c = c.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * N + offs_cn[None, :]
    c_mask = offs_cm[:, None] < M and offs_cn[None, :] < N
    tl.store(c_ptrs, c, mask=c_mask)


def tensordot_rsqrt(a: torch.Tensor, b: torch.Tensor, dims) -> torch.Tensor:
    a_dim = list(range(a.ndim))
    b_dim = list(range(b.ndim))
    if isinstance(dims, int):
        dims = [dims, dims]
    elif isinstance(dims, tuple):
        assert len(dims) == 2, "The length of dims must be 2"
    else:
        assert len(dims) == 2, "The length of dims must be 2"

    for dim in dims[0]:
        assert dim in a_dim, "Dimension out of range"
        assert a_dim.count(dim) == 1, "dim {} appears multiple times in the list of dims".format(dim)
    for dim in dims[1]:
        assert dim in b_dim, "Dimension out of range"
        assert b_dim.count(dim) == 1, "dim {} appears multiple times in the list of dims".format(dim)

    for dim in dims[0]:
        a_dim.remove(dim)
    for dim in dims[1]:
        b_dim.remove(dim)

    assert len(a_dim) == len(b_dim) > 0, "The number of dimensions of the outputs and inputs do not match"

    a_shape = list(a.shape)
    b_shape = list(b.shape)

    for dim in a_dim:
        assert a_shape[dim] == 1, "Broadcasting is currently not supported"
    for dim in b_dim:
        assert b_shape[dim] == 1, "Broadcasting is currently not supported"

    for i, dim in enumerate(a_dim):
        a_shape[dim] = a_shape[b_dim[i]]
        b_shape[b_dim[i]] = a_shape[b_dim[i]]

    o_shape = a_shape
    o_numels = 1
    for s in o_shape:
        o_numels *= s

    a_strides = list(a.stride())
    b_strides = list(b.stride())

    for i, dim in enumerate(a_dim):
        a_strides[dim] = 0
        b_strides[b_dim[i]] = 0

    a = a.reshape(a_shape).stride(a_strides)
    b = b.reshape(b_shape).stride(b_strides)

    c = torch.empty(o_numels, dtype=a.dtype, device=a.device)
    c_strides = list(c.stride())
    for i, dim in enumerate(a_dim):
        c_strides[dim] = 0
    c = c.reshape(a_shape).stride(c_strides)

    M = 1
    N = 1
    K = 1
    for m in a_shape:
        M *= m
    for n in b_shape:
        N *= n
    for k in a_shape:
        K *= k

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE_M"]), triton.cdiv(N, meta["BLOCK_SIZE_N"]))

    _tensordot_rsqrt_kernel[grid](a, b, c, M, N, K)
    return c
