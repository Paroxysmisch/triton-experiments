import triton
import triton.language as tl

@triton.jit
def _tensordot_kernel(
    A_ptr, B_ptr, C_ptr,
    A_shape, B_shape, C_shape,
    A_strides, B_strides, C_strides,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ACC_TYPE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    row = pid % M
    col = pid // M

    acc = tl.zeros((BLOCK_SIZE,), dtype=ACC_TYPE)

    for k in range(tl.cdiv(K, BLOCK_SIZE)):
        a_idx = row * A_strides[0] + k * A_strides[2]
        b_idx = k * B_strides[1] + col * B_strides[3]

        a = tl.load(A_ptr + a_idx, mask=a_idx < A_shape[0])
        b = tl.load(B_ptr + b_idx, mask=b_idx < B_shape[1])

        acc += a * b

    c_idx = row * C_strides[0] + col * C_strides[2]
    tl.store(C_ptr + c_idx, acc, mask=c_idx < C_shape[0])
