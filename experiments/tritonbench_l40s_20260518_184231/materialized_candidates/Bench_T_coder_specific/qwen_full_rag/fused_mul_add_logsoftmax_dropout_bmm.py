import torch
import triton
import triton.language as tl

@triton.jit
def fused_mul_add_logsoftmax_dropout_bmm_kernel(
    input1_ptr, input2_ptr, other_ptr, mat2_ptr, Y_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_input1_M, stride_input1_K,
    stride_input2_K, stride_input2_N,
    stride_other_M, stride_other_K,
    stride_mat2_B, stride_mat2_M, stride_mat2_N,
    stride_Y_B, stride_Y_M, stride_Y_N,
    bmm_algo: tl.constexpr, dropout_p: tl.constexpr,
    training: tl.constexpr, inplace: tl.constexpr,
    block_size: tl.constexpr, dot_dtype: tl.constexpr,
    HAS_OTHER: tl.constexpr, DO_DROPOUT: tl.constexpr,
    IS_BMM_HARDWARE_ACCELERATED: tl.constexpr,
    BLOCK_SIZE_SOFTMAX: tl.constexpr,
    BLOCK_SIZE_DROPOUT: tl.constexpr,
    BLOCK_SIZE_BMM: tl.constexpr,
):
    pid_B = tl.program_id(axis=1)
    pid_M = tl.program_id(axis=0)
    grid_N = tl.cdiv(N, BLOCK_SIZE_BMM)
    pid_N = tl.arange(0, BLOCK_SIZE_DROPOUT)[:, None] < grid_N
    pid_N = pid_N.to(tl.int32).to(BLOCK_SIZE_DROPOUT)[None, :] \
        + tl.arange(0, BLOCK_SIZE_DROPOUT)[None, :] \
        + pid_N * BLOCK_SIZE_DROPOUT
    pid_N = pid_N.to(tl.int32)

    input1_ptrs = input1_ptr \
        + pid_B * stride_input1_M \
        + (pid_M * block_size + tl.arange(0, block_size))[:, None] * stride_input1_M \
        + tl.arange(0, K) * stride_input1_K
    input1 = tl.load(input1_ptrs, mask=(pid_M * block_size + tl.arange(0, block_size)[:, None] \
        < M) & (tl.arange(0, K)[None, :] < K), other=0.0)
    input1 = input1.to(dot_dtype)

    input2_ptrs = input2_ptr \
        + pid_B * stride_input2_N \
        + tl.arange(0, K) * stride_input2_K \
        + (pid_N * BLOCK_SIZE_BMM + tl.arange(0, BLOCK_SIZE_BMM)) * stride_input2_N
    input2 = tl.load(input2_ptrs, mask=(tl.arange(0, K)[None, :] < K) \
        & ((pid_N * BLOCK_SIZE_BMM + tl.arange(0, BLOCK_SIZE_BMM)) < N), other=0.0)
    input2 = input2.to(dot_dtype)

    mul = input1 * input2

    if HAS_OTHER:
        other_ptrs = other_ptr \
            + pid_B * stride_other_M \
            + (pid_M * block_size + tl.arange(0, block_size)) * stride_other_M \
            + tl.arange(0, K) * stride_other_K
        other = tl.load(other_ptrs, mask=(pid_M * block_size + tl.arange(0, block_size)[:, None] \
            < M) & (tl.arange(0, K)[None, :] < K), other=0.0)
        other = other.to(dot_dtype)
        mul_add = mul + other
    else:
        mul_add = mul

    if DO_DROPOUT:
        dropout_mask = tl.rand(tl.float32, (BLOCK_SIZE_SOFTMAX, )) > dropout_p
        mul_add = tl.where(dropout_mask.to(mul_add.dtype), mul_add * (1.0 / (1.0 - dropout_p)), 0.0)

    BLOCK_SIZE_LIMIT = (
        BLOCK_SIZE_SOFTMAX if not DO_DROPOUT else
        BLOCK_SIZE_BMM if IS_BMM_HARDWARE_ACCELERATED else
        BLOCK_SIZE_DROPOUT
    )
    if block_size <= BLOCK_SIZE_LIMIT:
        mul_add_block_ptr = tl.make_block_ptr(
            base=mul_add,
            shape=(block_size, ),
            strides=(1, ),
            offsets=(pid_M * block_size, ),
            block_shape=(BLOCK_SIZE_LIMIT, ),
            order=(0, )
        )
        softmax_output = tl.softmax(mul_add_block_ptr, axis=0)
        if inplace:
            mul_add_ptrs = input1_ptr \
                + pid_B * stride_input1_M \
                + (pid_M * block_size + tl.arange(0, BLOCK_SIZE_LIMIT)) * stride_input1_M \
                + tl.arange(0, K) * stride_input1_K
            mul_add_ptrs = mul_add_ptrs.to(tl.pointer_type(dot_dtype))
            tl.store(mul_add_ptrs, softmax_output.to(dot_dtype), mask=((pid_M * block_size + tl.arange(0, BLOCK_SIZE_LIMIT)) < M) & (tl.arange(0, K)[None, :] < K))
        else:
            Y = mul_add.to(tl.float32)
            Y = Y - tl.max(Y, axis=0)
            numerator = tl.exp(Y)
            denominator = tl.sum(numerator, axis=0)

            Y_block_ptr = tl.make_block_ptr(
                base=Y,
                shape=(block_size, K),
                strides=(stride_Y_M, stride_Y_N),
                offsets=(pid_M * block_size, 0),
                block_shape=(BLOCK_SIZE_LIMIT, K),
                order=(1, 0)
            )
            denominator = tl.full((K, ), denominator, dtype=Y.dtype)
            denominator_block_ptr = tl.make_block_ptr(
                base=denominator,
                shape=(K, ),
                strides=(1, ),
                offsets=(0, ),
                block_shape=(K, ),
                order=(0, )
            )
            tl.atomic_add(Y_block_ptr, -tl.log(denominator_block_ptr))

            Y_ptrs = Y_ptr \
                + pid_B * stride_Y_B \
                + (pid_M * block_size + tl.arange(0, BLOCK_SIZE_LIMIT)) * stride_Y_M \
                + tl.arange(0, K) * stride_Y_N
            Y_ptrs = Y_ptrs.to(tl.pointer_type(dot_dtype))
            tl.store(Y_ptrs, Y.to(dot_dtype), mask=((pid_M * block_size + tl.arange(0, BLOCK_SIZE_LIMIT)) < M) & (tl.arange(0, K)[None, :] < K))

    if not IS_BMM_HARDWARE_ACCELERATED:
        mat2_ptrs = mat2_ptr \
            + pid_B * stride_mat2_B \
            + (pid_M * block_size + tl.arange(0, BLOCK_SIZE_BMM)) * stride_mat2_M \
            + (pid_N * BLOCK_SIZE_BMM + tl.arange(0, BLOCK_SIZE_BMM)) * stride_mat2_N
        mat2 = tl.load(mat2_ptrs, mask=((pid_M * block_size + tl.arange(0, BLOCK_SIZE_BMM)) < M) \
            & ((pid_N * BLOCK_SIZE_BMM + tl.arange(0, BLOCK_SIZE_BMM)) < N), other=0.0)
        mat2 = mat2.to(dot_dtype)

        Y = tl.dot(mul_add, mat2, allow_tf32=False)
    else:
        Y = mul_add

    Y_ptrs = Y_ptr \
        + pid_B * stride_Y_B \
        + (pid_M * block_size + tl.arange(0, BLOCK_SIZE_BMM)) * stride_Y_M \
        + (pid_N * BLOCK_SIZE_BMM + tl.arange(0, BLOCK_SIZE_BMM)) * stride_Y_N
    Y_ptrs = Y_ptrs.to(tl.pointer_type(dot_dtype))
    tl.store(Y_ptrs, Y.to(dot_dtype), mask=((pid_M * block_size + tl.arange(0, BLOCK_SIZE_BMM)) < M) & ((pid_N * BLOCK_SIZE_BMM + tl.arange(0, BLOCK_SIZE_BMM)) < N))


def fused_mul_add_logsoftmax_dropout_bmm(
    input1, input2, other, mat2, p=0.5, training=True, inplace=False, dim=-1, *, out=None
) -> torch.Tensor:
    finfo = torch.finfo(input1.dtype)
    dtype = input1.dtype
    assert input1.dtype in (torch.float16, torch.bfloat16, torch.float32) and input2.dtype == dtype \
        and other.dtype == dtype and mat2.dtype == dtype, "All tensors must have the same dtype"
    assert input1.shape[-1] == mat2.shape[-2] and input2.shape[-1] == mat2.shape[-1], "Incompatible dimensions"
    assert other.shape == () or other.shape == input1.shape, "The shapes of the `other` tensor and `input1` tensor must be broadcastable"
    assert mat2.is_contiguous(), "Matrix 2 must be contiguous"
    if out is None:
        Y = torch.empty_like(input2, dtype=dtype, device=input2.device)
    else:
        assert out.shape == input2.shape and out.dtype == dtype and out.device.type == input2.device.type, "Output tensor must have the same shape, dtype, and device as `input2` tensor"
        Y = out
    if dim < 0:
        dim = dim + input1.ndim + 1
    assert 0 <= dim <= input1.ndim, "Invalid dim"

    input1 = input1.contiguous()
    input2 = input2.contiguous()
    other = other.contiguous() if other.ndim > 0
