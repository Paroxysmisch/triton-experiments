import torch
import triton
import triton.language as tl
from .utils import get_config, broadcast_shapes

@triton.jit
def _tensordot_kernel(
    a_ptr, b_ptr, c_ptr, a_dims, b_dims, a_broadcasted_dims, b_broadcasted_dims, n_iters,
    reduction_stride_a, reduction_stride_b, out_offset, M, N, K, CACHE_KEY_M, CACHE_KEY_N,
    CACHE_KEY_K: tl.constexpr, BLOCK_SIZE_REDUCTION: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(CACHE_KEY_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(CACHE_KEY_N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, (pid + 1) // num_pid_in_group - first_pid_m)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=a_dims,
        strides=a_broadcasted_dims,
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=b_dims,
        strides=b_broadcasted_dims,
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(K, BLOCK_SIZE_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, n_iters):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    c = acc.to(tl.float16)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + out_offset + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


def _wrapper_tensordot(a: torch.Tensor, b: torch.Tensor, dims: Union[int, tuple, list]) -> torch.Tensor:
    device = a.device
    if a.device != b.device:
        if a.device.type == "cuda":
            b = b.to(device=a.device)
        else:
            a = a.to(device=b.device)

    a_dim = a.dim()
    b_dim = b.dim()

    if isinstance(dims, int):
        d = dims
        if d < 0:
            d += a_dim

        if d < 0 or d >= a_dim:
            raise ValueError(f"Dimension out of range (expected to be in range of [{-a_dim}, {a_dim - 1}], but got {d})")

        a_shape = broadcast_shapes([a.shape[:d], b.shape[d:]])
        b_shape = broadcast_shapes([a.shape[d:], b.shape[:d]])

        a_strides = [0] * a_dim
        for idx, size in enumerate(a_shape):
            a_strides[idx] = a.stride(idx)

        b_strides = [0] * b_dim
        for idx, size in enumerate(b_shape):
            b_strides[idx] = b.stride(idx)

        a = a.reshape(a_shape).as_strided(a_shape, a_strides)
        b = b.reshape(b_shape).as_strided(b_shape, b_strides)

        res_shape = list(a.shape)[:-1] + list(b.shape)[1:]
        res = torch.empty(res_shape, device=device, dtype=torch.float16)

        K = a.shape[d]
        M = a.numel() // K
        N = b.numel() // K

        grid_fn = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)
        _tensordot_kernel[grid_fn](a, b, res, a.shape, b.shape, a_shape, b_shape, N, a.stride(0), b.stride(0),
                                   res_offset=0, M=M, N=N, K=K, stride_cm=res.stride(0), stride_cn=res.stride(1),
                                   CACHE_KEY_M=M, CACHE_KEY_N=N, CACHE_KEY_K=K)
        return res
    elif isinstance(dims, (tuple, list)):
        if len(dims) != 2:
            raise ValueError("If given as a list, dims must have length 2")

        if all(isinstance(d, int) for d in dims):
            d0, d1 = dims
            if d0 < 0:
                d0 += a_dim
            if d1 < 0:
                d1 += b_dim

            if d0 < 0 or d0 >= a_dim:
                raise ValueError(
                    f"Dimension out of range (expected to be in range of [{-a_dim}, {a_dim - 1}], but got {d0})")

            if d1 < 0 or d1 >= b_dim:
                raise ValueError(
                    f"Dimension out of range (expected to be in range of [{-b_dim}, {b_dim - 1}], but got {d1})")

            a_shape = broadcast_shapes([a.shape[:d0], b.shape[d1:]])
            b_shape = broadcast_shapes([a.shape[d0:], b.shape[:d1]])

        elif all(isinstance(sublist, (tuple, list)) for sublist in dims):
            if len(dims[0]) != len(dims[1]):
                raise ValueError("The given dimension lists must be of equal length")

            if all(isinstance(el, int) for el in dims[0]) and all(isinstance(el, int) for el in dims[1]):
                d0, d1 = dims
                if any(d < 0 for d in d0):
                    d0 = [d if d >= 0 else d + a_dim for d in d0]
                if any(d < 0 for d in d1):
                    d1 = [d if d >= 0 else d + b_dim for d in d1]

                if any(d < 0 or d >= a_dim for d in d0):
                    raise ValueError(
                        f"Some dimension in {dims[0]} is out of range (expected to be in range of [{-a_dim}, {a_dim - 1}])")

                if any(d < 0 or d >= b_dim for d in d1):
                    raise ValueError(
                        f"Some dimension in {dims[1]} is out of range (expected to be in range of [{-b_dim}, {b_dim - 1}])")

                a_shape = broadcast_shapes([list(torch._indirect_select(a.shape, d0)), torch._indirect_select(b.shape, d1)])
                b_shape = broadcast_shapes([torch._indirect_select(a.shape, d1), torch._indirect_select(b.shape, d0)])

            else:
                raise ValueError("Both elements of `dim` must either be ints or iterables of ints")
        else:
            raise ValueError("Elements of `dim` must all be of the same type")

        a_strides = [0] * a_dim
        for idx, size in enumerate(a_shape):
            a_strides[idx] = a.stride(idx)

        b_strides = [0] * b_dim
        for idx, size in enumerate(b_shape):
            b_strides[idx] = b.stride(idx)

        a = a.reshape(a_shape).as_strided(a_shape, a_strides)
        b = b.reshape(b_shape).as_strided(b_shape, b_strides)

        res_shape = list(a.shape)[:-1] + list(b.shape)[1:]
        res = torch.empty(res_shape, device=device, dtype=torch.float16)

        K = a.shape[d0] if isinstance(d0, int) else a.shape[len(a_shape) - len(d0)]
        M = a.numel() // K
        N = b.numel() // K

        grid_fn = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)
        _tensordot_kernel[grid_fn](a, b, res, a.shape, b.shape, a_shape, b_shape, N, a.stride(0), b.stride(0),
                                   res_offset=0, M=M, N=N, K=K, stride_cm=res.stride(0), stride_cn=res.stride(1),
                                   CACHE_KEY_M=M, CACHE_KEY_N=N, CACHE_KEY_K=K)
        return res
