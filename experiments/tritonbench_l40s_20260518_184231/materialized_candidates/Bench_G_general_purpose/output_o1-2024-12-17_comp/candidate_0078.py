import triton
import triton.language as tl

@triton.jit
def _int8_matmul_rowwise_dequantize(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    strideAm, strideAk,
    strideBk, strideBn,
    strideCm, strideCn,
    state_x_ptr, state_w_ptr, bias_ptr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    row_idxs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_idxs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_begin = pid_k * BLOCK_K

    # Create pointers for A, B blocks
    a_ptrs = A_ptr + (row_idxs[:, None] * strideAm + (k_begin + tl.arange(0, BLOCK_K)[None, :]) * strideAk)
    b_ptrs = B_ptr + ((k_begin + tl.arange(0, BLOCK_K)[:, None]) * strideBk + col_idxs[None, :] * strideBn)

    # Load int8 values and convert to int32 for accumulation
    a_vals = tl.load(a_ptrs, mask=(row_idxs[:, None] < M) & (k_begin + tl.arange(0, BLOCK_K)[None, :] < K), other=0).to(tl.int32)
    b_vals = tl.load(b_ptrs, mask=(col_idxs[None, :] < N) & (k_begin + tl.arange(0, BLOCK_K)[:, None] < K), other=0).to(tl.int32)

    # Compute partial block dot-product
    acc = tl.dot(a_vals, b_vals)

    # For multiple K-splits, accumulate partial results in atomic fashion if needed
    # But first apply scale factors from state_x and state_w
    # state_x is typically per row of A, state_w is typically per row of B
    sx = tl.load(state_x_ptr + row_idxs, mask=row_idxs < M, other=1.0)
    sw = tl.load(state_w_ptr + col_idxs, mask=col_idxs < N, other=1.0)

    # Convert to float, scale row-wise (broadcast over each dimension)
    acc = acc.to(tl.float32)
    acc = acc * sx[:, None] * sw[None, :]

    # Optional bias
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + col_idxs, mask=col_idxs < N, other=0.0)
        acc = acc + bias[None, :]

    # Write out results
    c_ptrs = C_ptr + (row_idxs[:, None] * strideCm + col_idxs[None, :] * strideCn)
    if SPLIT_K > 1:
        # Accumulate using atomic adds if we are splitting K
        mask = (row_idxs[:, None] < M) & (col_idxs[None, :] < N)
        tl.atomic_add(c_ptrs, acc, mask=mask)
    else:
        # Write final results
        mask = (row_idxs[:, None] < M) & (col_idxs[None, :] < N)
        tl.store(c_ptrs, acc, mask=mask)


def int8_matmul_rowwise_dequantize(
    A, B, state_x, state_w, bias=None, split_k=1, block_m=128, block_n=128, block_k=32
):
    """
    A: 2D int8 tensor [M, K]
    B: 2D int8 tensor [K, N]
    state_x: 1D float tensor [M] for row-wise scaling of A
    state_w: 1D float tensor [N] for row-wise scaling of B
    bias: 1D float tensor [N] or None
    """
    assert A.is_contiguous()
    assert B.is_contiguous()
    assert state_x.is_contiguous()
    assert state_w.is_contiguous()

    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, "A.columns must match B.rows"
    if bias is not None:
        assert bias.shape[0] == N

    # Allocate output
    import torch
    C = torch.zeros((M, N), dtype=torch.float32, device=A.device)

    grid = lambda META: (
        ( (M + block_m - 1) // block_m ),
        ( (N + block_n - 1) // block_n ),
        split_k
    )

    # Strides
    strideAm = A.stride(0)
    strideAk = A.stride(1)
    strideBk = B.stride(0)
    strideBn = B.stride(1)
    strideCm = C.stride(0)
    strideCn = C.stride(1)

    _int8_matmul_rowwise_dequantize[grid](
        A, B, C,
        M, N, K,
        strideAm, strideAk,
        strideBk, strideBn,
        strideCm, strideCn,
        state_x, state_w, bias if bias is not None else tl.nullptr,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split_k
    )
    return C
