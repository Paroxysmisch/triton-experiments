import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    input_shape,
    weight_shape,
    bias_shape,
    stride,
    padding,
    dilation,
    groups,
    BLOCK_SIZE_I: tl.constexpr,
    BLOCK_SIZE_J: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_blocks_i = tl.cdiv(input_shape[2], BLOCK_SIZE_I)
    num_blocks_j = tl.cdiv(input_shape[3], BLOCK_SIZE_J)

    i = pid // num_blocks_j * stride + padding
    j = pid % num_blocks_j * stride + padding

    accum = tl.zeros((BLOCK_SIZE_K,), dtype=input_ptr.dtype)

    for k in range(0, weight_shape[2] + 2 * dilation - 1, BLOCK_SIZE_K):
        ii = i + k // dilation
        jj = j + k % dilation
        if ii < input_shape[2] and jj < input_shape[3]:
            # Load input and weight
            x = tl.load(input_ptr + ((pid // num_blocks_j * BLOCK_SIZE_I + i) * input_shape[3] + jj))
            w = tl.load(weight_ptr + ((pid % num_blocks_j * BLOCK_SIZE_I + j) * weight_shape[3] + k))
            accum += x * w

    if bias_ptr is not None:
        accum += tl.load(bias_ptr + pid)

    tl.store(output_ptr + pid, accum)
