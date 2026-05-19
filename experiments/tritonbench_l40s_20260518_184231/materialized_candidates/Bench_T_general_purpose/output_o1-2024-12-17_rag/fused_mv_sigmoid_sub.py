import torch
import triton
import triton.language as tl

@triton.jit
def _fused_mv_sigmoid_sub_kernel(
    A_ptr, V_ptr, Other_ptr, Out_ptr,
    N, M,
    strideAm,  # distance between rows in A
    strideA,   # distance between elements in A
    strideV,   # distance between elements in V
    strideOut, # distance between elements in Out
    alpha,
    BROADCAST_OTHER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Compute matrix-vector product
    # Each index in offsets is a row; compute the dot product with V
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for col in range(0, M):
        # Load A[row, col] if valid
        a_val = tl.load(A_ptr + offsets * strideAm + col * strideA, mask=mask, other=0.0)
        # Load V[col]
        v_val = tl.load(V_ptr + col * strideV)
        acc += a_val * v_val

    # Sigmoid
    acc = 1.0 / (1.0 + tl.exp(-acc))

    # Subtract alpha * other
    if BROADCAST_OTHER:
        # Other is a scalar/broadcast
        other_val = tl.load(Other_ptr)
        acc = acc - alpha * other_val
    else:
        # Other has
