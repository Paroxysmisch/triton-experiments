import triton
import triton.language as tl

@triton.jit
def tensordot_kernel(
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    c_ptr: tl.tensor,
    a_shape: tl.constexpr,
    b_shape: tl.constexpr,
    c_shape: tl.constexpr,
    a_strides: tl.constexpr,
    b_strides: tl.constexpr,
    c_strides: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    block_size_m: tl.constexpr,
    block_size_n: tl.constexpr,
    block_size_k: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(m, block_size_m)
    grid_n = tl.cdiv(n, block_size_n)

    # Compute row and column indices within the block
    row_in_block = pid % grid_m
    col_in_block = pid // grid_m

    # Compute global row and column indices
    row_global = row_in_block * block_size_m + tl.arange(0, block_size_m)
    col_global = col_in_block * block_size_n + tl.arange(0, block_size_n)

    # Initialize the result to zero
    acc = tl.zeros((block_size_m, block_size_n), dtype=a.dtype)

    # Loop over the intermediate dimension k
    for k in range(0, d, block_size_k):
        k_in_block = tl.arange(0, block_size_k)
        a_slice = a_ptr[row_global[:, None] * a_strides[0] + k_in_block[None, :]]
        b_slice = b_ptr[k_in_block[:, None] * b_strides[2] + col_global]
        acc += a_slice * b_slice

    # Write the result back to global memory
    c_ptr[row_global[:, None] * c_strides[0] + col_global] = acc

@triton.autotune(
    configs=[
        triton.Config({'block_size_m': 32, 'block_size_n': 32, 'block_size_k': 8}, num_stages=2, num_warps=4),
        triton.Config({'block_size_m': 64, 'block_size_n': 64, 'block_size_k': 16}, num_stages=2, num_warps=4),
    ],
    key=['m', 'n', 'd']
)
def tensordot(
    a: tl.Tensor,
    b: tl.Tensor,
    dims: int or tuple[list[int], list[int]],
) -> tl.Tensor:
    if isinstance(dims, int):
        d = dims
        m = len(a.shape) - d
        n = len(b.shape) - d
        assert a.shape[m:] == b.shape[:d], "Contracted dimensions do not match"
        a_shape = a.shape[:m] + b.shape[d:]
        b_shape = a.shape[m:d] + b.shape[d:]
    elif isinstance(dims, tuple):
        dim_a, dim_b = dims
        m = len(dim_a)
        n = len(dim_b)
        assert set(dim_a).issubset(range(len(a.shape))), "Invalid dimension index for a"
        assert set(dim_b).issubset(range(len(b.shape))), "Invalid dimension index for b"
        assert all(a.shape[i] == b.shape[j] for i, j in zip(dim_a, dim_b)), "Contracted dimensions do not match"
        a_shape = [a.shape[i] for i in range(len(a.shape)) if i not in dim_a]
        b_shape = [b.shape[j] for j in range(len(b.shape)) if j not in dim_b]
    else:
        raise ValueError("Unsupported type for dims")

    c_shape = a_shape + b_shape[len(a_shape):]

    # Allocate output tensor
    c = tl.zeros(c_shape, dtype=a.dtype)

    # Set strides
    a_strides = tl.stride(a, a_shape)
    b_strides = tl.stride(b, b_shape)
    c_strides = tl.stride(c, c_shape)

    # Call the kernel
    tensordot_kernel[
        grid=(tl.cdiv(m, 32), tl.cdiv(n, 32)),
        block=(32, 32, 1),
        num_warps=4,
    ](a.data, b.data, c.data, a_shape, b_shape, c_shape, a_strides, b_strides, c_strides, m, n, d, 32, 32, 8)

    return c
