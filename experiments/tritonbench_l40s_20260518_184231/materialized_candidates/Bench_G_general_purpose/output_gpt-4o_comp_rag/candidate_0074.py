import triton
import triton.language as tl
import torch

@triton.jit
def _int8_matmul_rowwise_dequantize_kernel(
    A_ptr, B_ptr, C_ptr,
    state_x_ptr, state_w_ptr,
    bias_ptr, has_bias: tl.constexpr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr
):
    # Compute block indices
    pid = tl.program_id(axis=0)
    pid_z = pid // (M // BLOCK_M)
    pid_y = (pid % (M // BLOCK_M)) // (N // BLOCK_N)
    pid_x = pid % (N // BLOCK_N)

    # Compute start of the block
    offs_am = pid_y * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_x * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load blocks of A and B
        a = tl.load(A_ptr + (offs_am[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak))
        b = tl.load(B_ptr + ((k + offs_k)[:, None] * stride_bk + offs_bn[None, :] * stride_bn))

        # Compute dot product
        acc += tl.dot(a, b)

    # Dequantize
    scale_x = tl.load(state_x_ptr + offs_am)
    scale_w = tl.load(state_w_ptr + offs_bn)
    acc = acc.to(tl.float32) * scale_x[:, None] * scale_w[None, :]

    # Add bias if present
    if has_bias:
        bias = tl.load(bias_ptr + offs_bn)
        acc += bias[None, :]

    # Store result
    c = acc.to(tl.float32)
    if SPLIT_K > 1:
        # Atomic add for partial results
        tl.atomic_add(C_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn), c)
    else:
        # Direct store
        tl.store(C_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn), c)

def int8_matmul_rowwise_dequantize(
    A, B, state_x, state_w, bias=None, SPLIT_K=1,
    BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
):
    assert A.dtype == torch.int8 and B.dtype == torch.int8
    M, K = A.shape
    K, N = B.shape

    # Allocate output
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)

    # Define grid
    grid = lambda META: (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    # Launch kernel
    _int8_matmul_rowwise_dequantize_kernel[grid](
        A, B, C,
        state_x, state_w,
        bias if bias is not None else torch.tensor(0, device=A.device),
        bias is not None,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K
    )
    return C
