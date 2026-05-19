import triton.language as tl

@triton.jit
def _int8_matmul_rowwise_dequantize(
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    A_ptr,
    B_ptr,
    C_ptr,
    bias_ptr,
    state_x_ptr,
    state_w_ptr,
    M,
    N,
    K,
    LD_A,
    LD_B,
    LD_C,
    stride_x,
    stride_y,
    out_dtype: tl.constexpr,
    compute_dtype: tl.constexpr,
    has_bias: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Calculate the start indices and offsets for loading and storing data
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N
    k_start = tl.program_id(axis=2) * SPLIT_K

    # Perform the dot product
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k in range(0, K, SPLIT_K):
        k_end = min(k + SPLIT_K, K)

        # Load input data and scale factors
        x = tl.load(state_x_ptr + row_start + k * stride_x)
        w = tl.reduce_max(tl.load(state_w_ptr + col_start + k * stride_y))

        for kk in range(k, k_end):
            a = tl.load(A_ptr + row_start + kk * LD_A)
            b = tl.load(B_ptr + col_start + kk * LD_B)
            acc += tl.dot(a, b, out_dtype=compute_dtype)

    # Perform scale and bias if present
    acc = acc * x * w

    if has_bias:
        bias = tl.load(bias_ptr + pid_n)
        acc += bias

    # Store the result
    C_offset = row_start * LD_C + col_start
    tl.store(C_ptr + C_offset, acc)

def int8_matmul_rowwise_dequantize(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    bias: torch.Tensor,
    state_x: torch.Tensor,
    state_w: torch.Tensor,
):
    # Prepare the tensors for input
    M, N, K = C.shape
    LD_A, LD_B, LD_C = A.stride(), B.stride(), C.stride()

    # Run the Triton kernel
    _int8_matmul_rowwise_dequantize[M, N](
        A.contiguous().data_ptr(),
        B.contiguous().data_ptr(),
        C.contiguous().data_ptr(),
        bias.contiguous().data_ptr(),
        state_x.contiguous().data_ptr(),
        state_w.contiguous().data_ptr(),
        M, N, K,
        LD_A, LD_B, LD_C,
        state_x.stride(0),
        state_w.stride(0),
        torch.int32,
        torch.float32,
        bias is not None
    )
