import triton
import triton.language as tl

# Triton kernel for the forward pass of softmax
@triton.jit
def _softmax(
    X,  # input tensor
    L,  # optional mask (for causal or other types of masking)
    Y,  # output tensor
    stride_xm, stride_xn,  # strides for input tensor
    stride_ym, stride_yn,  # strides for output tensor
    stride_lm, stride_ln,  # strides for mask tensor (if used)
    M, N,  # dimensions of the tensor
    IS_FP16: tl.constexpr,  # flag for half-precision input
    LOG: tl.constexpr,  # flag for log-softmax
    CAUSAL: tl.constexpr,  # flag for causal masking
    MASK_TYPE: tl.constexpr,  # type of mask (if any)
    BLOCK_SIZE: tl.constexpr  # block size for parallelism
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets_m = block_start + tl.arange(0, BLOCK_SIZE)
    offsets_n = tl.arange(0, N)

    X_block_ptr = X + offsets_m[:, None] * stride_xm + offsets_n[None, :] * stride_xn
    L_block_ptr = L + offsets_m[:, None] * stride_lm + offsets_n[None, :] * stride_ln if L is not None else None
    Y_block_ptr = Y + offsets_m[:, None] * stride_ym + offsets_n[None, :] * stride_yn

    X_block = tl.load(X_block_ptr, mask=offsets_m[:, None] < M, other=-float('inf'))

    if L is not None:
        L_block = tl.load(L_block_ptr, mask=offsets_m[:, None] < M, other=0.0)
        X_block += L_block

    if CAUSAL:
        mask = (offsets_m[:, None] >= offsets_n[None, :])
        X_block = tl.where(mask, X_block, -float('inf'))

    max_val = tl.max(X_block, 1)
    max_val = max_val[:, None]
    X_block = X_block - max_val

    if IS_FP16:
        X_block = X_block.to(tl.float32)

    exp_X_block = tl.exp(X_block)

    sum_exp = tl.sum(exp_X_block, 1)
    sum_exp = sum_exp[:, None]

    if LOG:
        Y_block = X_block - tl.log(sum_exp)
    else:
        Y_block = exp_X_block / sum_exp

    if IS_FP16:
        Y_block = Y_block.to(tl.float16)

    tl.store(Y_block_ptr, Y_block, mask=offsets_m[:, None] < M)

# Host function for the forward pass of softmax
def softmax(X, L=None, LOG=False, CAUSAL=False, MASK_TYPE=None):
    M, N = X.shape[0], X.shape[2]
    Y = tl.zeros_like(X)

    stride_xm, stride_xn = X.stride(0), X.stride(2)
    stride_ym, stride_yn = Y.stride(0), Y.stride(2)
    stride_lm, stride_ln = L.stride(0), L.stride(2) if L is not None else (0, 0)

    IS_FP16 = X.dtype == tl.float16

    grid = (M,)

    _softmax[grid](
        X, L, Y,
        stride_xm, stride_xn,
        stride_ym, stride_yn,
        stride_lm, stride_ln,
        M, N,
        IS_FP16, LOG, CAUSAL, MASK_TYPE,
        BLOCK_SIZE=triton.next_power_of_2(N)
    )

    return Y

# Triton kernel for the backward pass of softmax
@triton.jit
def _softmax_backward(
    dY,  # gradient of the output
    Y,  # output of the forward pass
    dX,  # gradient of the input
    stride_dym, stride_dyn,  # strides for gradient of the output
    stride_ym, stride_yn,  # strides for output of the forward pass
    stride_dxm, stride_dxn,  # strides for gradient of the input
    M, N,  # dimensions of the tensor
    LOG: tl.constexpr,  # flag for log-softmax
    CAUSAL: tl.constexpr,  # flag for causal masking
    BLOCK_SIZE: tl.constexpr  # block size for parallelism
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets_m = block_start + tl.arange(0, BLOCK_SIZE)
    offsets_n = tl.arange(0, N)

    dY_block_ptr = dY + offsets_m[:, None] * stride_dym + offsets_n[None, :] * stride_dyn
    Y_block_ptr = Y + offsets_m[:, None] * stride_ym + offsets_n[None, :] * stride_yn
    dX_block_ptr = dX + offsets_m[:, None] * stride_dxm + offsets_n[None, :] * stride_dxn

    dY_block = tl.load(dY_block_ptr, mask=offsets_m[:, None] < M, other=0.0)
    Y_block = tl.load(Y_block_ptr, mask=offsets_m[:, None] < M, other=0.0)

    if CAUSAL:
        mask = (offsets_m[:, None] >= offsets_n[None, :])
        dY_block = tl.where(mask, dY_block, 0.0)

    sum_dY = tl.sum(dY_block, 1)
    sum_dY = sum_dY[:, None]

    if LOG:
        dX_block = dY_block - Y_block * sum_dY
    else:
        dX_block = Y_block * (dY_block - Y_block * sum_dY)

    tl.store(dX_block_ptr, dX_block, mask=offsets_m[:, None] < M)

# Host function for the backward pass of softmax
def softmax_backward(dY, Y, LOG=False, CAUSAL=False):
    M, N = dY.shape[0], dY.shape[2]
    dX = tl.zeros_like(dY)

    stride_dym, stride_dyn = dY.stride(0), dY.stride(2)
    stride_ym, stride_yn = Y.stride(0), Y.stride(2)
    stride_dxm, stride_dxn = dX.stride(0), dX.stride(2)

    grid = (M,)

    _softmax_backward[grid](
        dY, Y, dX,
        stride_dym, stride_dyn,
        stride_ym, stride_yn,
        stride_dxm, stride_dxn,
        M, N,
        LOG, CAUSAL,
        BLOCK_SIZE=triton.next_power_of_2(N)
    )

    return dX
