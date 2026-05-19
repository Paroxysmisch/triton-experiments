import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ra = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    rb = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Pointer arithmetic
    A_ptrs = A + (rm[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak)
    B_ptrs = B + (tl.arange(0, BLOCK_K)[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Loop over K dimension
    for k_off in range(0, K, BLOCK_K):
        k_mask = k_off + tl.arange(0, BLOCK_K) < K
        a_val = tl.where(k_mask[None, :], tl.load(A_ptrs, mask=k_mask[None, :], other=0.), 0.)
        b_val = tl.where(k_mask[:, None], tl.load(B_ptrs, mask=k_mask[:, None], other=0.), 0.)
        ra = a_val.to(tl.float32)
        rb = b_val.to(tl.float32)
        accum += tl.dot(ra, rb)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Write back
    c_mask_m = rm < M
    c_mask_n = rn < N
    tl.store(
        C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn),
        accum,
        mask=c_mask_m[:, None] & c_mask_n[None, :]
    )

def matmul(input, other, *, out=None):
    # 1D & 1D => Dot product (no out)
    if input.dim() == 1 and other.dim() == 1:
        return torch.sum(input * other)

    # Expand dims for broadcasting if needed
    max_dim = max(input.dim(), other.dim())
    input_exp = input.reshape((1,) * (max_dim - input.dim()) + tuple(input.shape))
    other_exp = other.reshape((1,) * (max_dim - other.dim()) + tuple(other.shape))

    # If last two dims are 1D x 2D or 2D x 1D => treat as 2D x 2D, or handle vector multiplication
    # For general N>2, we'll do a batched approach by flattening leading dimensions
    # into a single batch dimension and reshaping back after Triton execution.

    # Flatten leading batch dims if any
    batch_shape = input_exp.shape[:-2]
    M1, K1 = input_exp.shape[-2:]
    M2, K2 = other_exp.shape[-2:]
    if K1 != M2:
        # Broadcasting check or raise
        # We let PyTorch style broadcasting handle it, or raise if shapes are incompatible
        broadcasted = torch.broadcast_shapes(input_exp.shape, other_exp.shape)
        input_exp = input_exp.expand(broadcasted)
        other_exp = other_exp.expand(broadcasted)
        batch_shape = broadcasted[:-2]
        M1, K1 = input_exp.shape[-2:]
        M2, K2 = other_exp.shape[-2:]
        if K1 != M2:
            raise RuntimeError("Shapes not compatible for matmul")

    # Reshape to 2D if there's a batch dimension
    batch_size = 1
    for s in batch_shape:
        batch_size *= s
    A_2d = input_exp.reshape(batch_size, M1, K1)
    B_2d = other_exp.reshape(batch_size, M2, K2)

    # Prepare output
    out_shape = batch_shape + (M1, K2)
    if out is None:
        C = input.new_empty(out_shape)
    else:
        if out.shape != out_shape:
            raise RuntimeError("The out tensor has the wrong shape")
        C = out

    # Launch Triton kernel over each batch
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    for b in range(batch_size):
        A_curr = A_2d[b]
        B_curr = B_2d[b]
        C_curr = C[b] if batch_size > 1 else C

        M = A_curr.size(0)
        K = A_curr.size(1)
        N = B_curr.size(1)

        # Strides
        stride_am = A_curr.stride(0)
        stride_ak = A_curr.stride(1)
        stride_bk = B_curr.stride(0)
        stride_bn = B_curr.stride(1)
        stride_cm = C_curr.stride(0)
        stride_cn = C_curr.stride(1)

        grid = (
            ( (M + BLOCK_M - 1) // BLOCK_M ),
            ( (N + BLOCK_N - 1) // BLOCK_N )
        )

        triton.run(
            _matmul_kernel,
            grid=grid,
            num_warps=4,
            num_stages=2,
            args=[
                A_curr, B_curr, C_curr,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn
            ],
            kwargs={
                "BLOCK_M": BLOCK_M,
                "BLOCK_N": BLOCK_N,
                "BLOCK_K": BLOCK_K
            }
        )

    if len(batch_shape) > 0:
        return C.reshape(out_shape)
    else:
        return C.squeeze(0) if out is None else out.squeeze(0)
