import triton
import triton.language as tl
import torch

def _next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2**((x - 1).bit_length())

@triton.jit
def _dot_product_kernel(
    x_ptr, 
    y_ptr, 
    out_ptr, 
    N: tl.constexpr, 
    block_size: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    result = tl.sum(x * y, axis=0)

    if pid == 0:
        tl.store(out_ptr, result)

@triton.jit
def _matmul_kernel(
    A_ptr, 
    B_ptr, 
    C_ptr, 
    M: tl.constexpr, 
    N: tl.constexpr, 
    K: tl.constexpr, 
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm_mask = rm < M
    rn_mask = rn < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in BLOCK_K chunks
    for k in range(0, K, BLOCK_K):
        k_range = k + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        # Load A and B
        a = tl.load(
            A_ptr + (rm[:, None] * K + k_range[None, :]),
            mask=(rm_mask[:, None] & k_mask[None, :]),
            other=0.0
        )
        b = tl.load(
            B_ptr + (k_range[:, None] * N + rn[None, :]),
            mask=(k_mask[:, None] & rn_mask[None, :]),
            other=0.0
        )

        acc += tl.dot(a, b)

    # Write back
    c = acc
    tl.store(
        C_ptr + (rm[:, None] * N + rn[None, :]),
        c,
        mask=(rm_mask[:, None] & rn_mask[None, :])
    )

def matmul(input: torch.Tensor, other: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # 1) Handle 1D x 1D => Dot product (no out supported)
    if input.dim() == 1 and other.dim() == 1:
        if out is not None:
            raise ValueError("The 1D dot product version does not support an out parameter.")
        if input.size(0) != other.size(0):
            raise ValueError("Input 1D tensors must be the same size for dot product.")

        N = _next_power_of_2(input.size(0))
        block_size = 1024
        result = torch.empty((), dtype=torch.float32, device=input.device)

        grid = (1,)
        _dot_product_kernel[grid](input, other, result, N, block_size)
        return result

    # 2) Handle >= 2D => Possibly matrix multiplication or batched
    # For simplicity, handle pure 2D or broadcasted cases
    if input.dim() == 2 and other.dim() == 2:
        # 2D x 2D => matrix-matrix product
        m, k1 = input.shape
        k2, n = other.shape
        if k1 != k2:
            raise ValueError("Inner dimensions must match for 2D matrix multiplication.")
        if out is not None:
            # Verify out shape
            if out.shape != (m, n):
                raise ValueError("Out tensor shape must match resulting matrix shape.")
            result = out
        else:
            result = torch.empty((m, n), dtype=input.dtype, device=input.device)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid = ((m + BLOCK_M - 1) // BLOCK_M, (n + BLOCK_N - 1) // BLOCK_N)
        _matmul_kernel[grid](
            input, other, result, 
            m, n, k1, 
            BLOCK_M, BLOCK_N, BLOCK_K
        )
        return result
    
    # 3) For 2D x 1D, 1D x 2D => matrix-vector or vector-matrix
    # We'll convert to 2D x 2D by reshaping the vector
    if input.dim() == 2 and other.dim() == 1:
        m, k = input.shape
        if k != other.size(0):
            raise ValueError("Inner dimensions must match for matrix-vector product.")
        other_2d = other.view(-1, 1)
        temp = matmul(input, other_2d, out=None)
        return temp.view(m)

    if input.dim() == 1 and other.dim() == 2:
        k, n = other.shape
        if input.size(0) != k:
            raise ValueError("Inner dimensions must match for vector-matrix product.")
        input_2d = input.view(1, -1)
        temp = matmul(input_2d, other, out=None)
        return temp.view(n)
    
    # 4) N-dimensional (N>2) => batched/broadcasted matmul
    # A simple approach using torch for actual broadcast, then Triton on each sub-batch
    # (not the most efficient, but demonstrates concept)
    # We'll flatten the leading dimensions into a single batch dimension, then unflatten
    # for the result. Broadcasting is handled by expanding input/other via torch.
    # Then we apply the 2D kernel on each batch slice.
    input_b = input.unsqueeze(-3) if input.dim() == 1 else input
    other_b = other.unsqueeze(-3) if other.dim() == 1 else other
    # Ensure at least 2D
    while input_b.dim() < 2:
        input_b = input_b.unsqueeze(0)
    while other_b.dim() < 2:
        other_b = other_b.unsqueeze(0)

    # Broadcast shapes
    broadcast_shape = torch.broadcast_shapes(input_b.shape[:-2], other_b.shape[:-2])
    a_expanded = input_b.expand(*broadcast_shape, *input_b.shape[-2:])
    b_expanded = other_b.expand(*broadcast_shape, *other_b.shape[-2:])

    # Flatten batch dims
    batch_size = 1
    for dim in broadcast_shape:
        batch_size *= dim
    M, K = a_expanded.shape[-2:]
    K2, N = b_expanded.shape[-2:]
    if K != K2:
        raise ValueError("Incompatible matrix dimensions in broadcasted matmul.")
    
    a_reshape = a_expanded.contiguous().view(batch_size, M, K)
    b_reshape = b_expanded.contiguous().view(batch_size, K, N)

    if out is not None:
        if out.shape != (*broadcast_shape, M, N):
            raise ValueError("Out tensor shape must match the broadcasted result shape.")
        c_reshape = out.view(batch_size, M, N)
    else:
        c_reshape = torch.empty((batch_size, M, N), dtype=input.dtype, device=input.device)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        (batch_size * ((M + BLOCK_M - 1) // BLOCK_M)),
        ((N + BLOCK_N - 1) // BLOCK_N),
    )

    # We'll combine batch index into pid_m offset: pid_m // num_block_m gives batch
    # We'll store the kernel arguments as well
    @triton.jit
    def _batched_matmul_kernel(
        A_ptr, B_ptr, C_ptr, 
        batch_size: tl.constexpr,
        M: tl.constexpr, 
        N: tl.constexpr, 
        K: tl.constexpr, 
        BLOCK_M: tl.constexpr, 
        BLOCK_N: tl.constexpr, 
        BLOCK_K: tl.constexpr
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        # extract batch from pid_m
        blocks_per_batch = (M + BLOCK_M - 1) // BLOCK_M
        batch_id = pid_m // blocks_per_batch

        row_block_id = pid_m % blocks_per_batch
        col_block_id = pid_n

        rm = row_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = col_block_id * BLOCK_N + tl.arange(0, BLOCK_N)

        rm_mask = rm < M
        rn_mask = rn < N

        # offset for batch
        A_batch_ptr = A_ptr + batch_id * (M * K)
        B_batch_ptr = B_ptr + batch_id * (K * N)
        C_batch_ptr = C_ptr + batch_id * (M * N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            k_rng = k + tl.arange(0, BLOCK_K)
            k_mask = k_rng < K

            a = tl.load(
                A_batch_ptr + (rm[:, None] * K + k_rng[None, :]),
                mask=(rm_mask[:, None] & k_mask[None, :]),
                other=0.0
            )
            b = tl.load(
                B_batch_ptr + (k_rng[:, None] * N + rn[None, :]),
                mask=(k_mask[:, None] & rn_mask[None, :]),
                other=0.0
            )
            acc += tl.dot(a, b)

        tl.store(
            C_batch_ptr + (rm[:, None] * N + rn[None, :]),
            acc,
            mask=(rm_mask[:, None] & rn_mask[None, :])
        )

    _batched_matmul_kernel[grid](
        a_reshape, 
        b_reshape, 
        c_reshape, 
        batch_size, 
        M, 
        N, 
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K
    )

    if out is not None:
        return out
    else:
        return c_reshape.view(*broadcast_shape, M, N)
