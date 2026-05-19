import triton
import triton.language as tl

@triton.jit
def rotary_kernel(
    X, COS, SIN, IS_VARLEN: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    INTERLEAVED: tl.constexpr, CONJUGATE: tl.constexpr, BATCH, HEAD, SEQ_LEN, HEAD_DIM
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(SEQ_LEN, BLOCK_M)
    num_pid_n = tl.cdiv(HEAD_DIM, BLOCK_K)
    num_pid_in_batch = num_pid_m * num_pid_n
    batch_id = pid // num_pid_in_batch
    pid = pid % num_pid_in_batch
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_K

    offsets_m = block_start_m + tl.arange(0, BLOCK_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_K)
    offsets = offsets_m[:, None] * HEAD_DIM + offsets_n[None, :]

    X_block_ptr = tl.make_block_ptr(
        base=X + batch_id * (SEQ_LEN * HEAD_DIM),
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(block_start_m, block_start_n),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0)
    )

    COS_block_ptr = tl.make_block_ptr(
        base=COS,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(block_start_m, block_start_n),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0)
    )

    SIN_block_ptr = tl.make_block_ptr(
        base=SIN,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(HEAD_DIM, 1),
        offsets=(block_start_m, block_start_n),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0)
    )

    x = tl.load(X_block_ptr)
    cos = tl.load(COS_block_ptr)
    sin = tl.load(SIN_block_ptr)

    if CONJUGATE:
        sin = -sin

    if INTERLEAVED:
        x = x.to(tl.float32)
        cos = cos.to(tl.float32)
        sin = sin.to(tl.float32)
        x_rotated = x * cos - tl.roll(x, shift=1, axis=1) * sin
    else:
        x_rotated = x * cos - tl.roll(x, shift=HEAD_DIM // 2, axis=1) * sin

    tl.store(X_block_ptr, x_rotated)

import torch
import triton
import triton.language as tl

def apply_rotary(X, COS, SIN, IS_VARLEN=False, BLOCK_M=128, BLOCK_K=128, INTERLEAVED=False, CONJUGATE=False):
    # Ensure tensors are contiguous
    X = X.contiguous()
    COS = COS.contiguous()
    SIN = SIN.contiguous()

    # Check data types
    assert X.dtype == torch.float32, "X must be of type torch.float32"
    assert COS.dtype == torch.float32, "COS must be of type torch.float32"
    assert SIN.dtype == torch.float32, "SIN must be of type torch.float32"

    # Get tensor dimensions
    BATCH, SEQ_LEN, HEAD, HEAD_DIM = X.shape

    # Calculate grid size
    grid = (BATCH * (tl.cdiv(SEQ_LEN, BLOCK_M) * tl.cdiv(HEAD_DIM, BLOCK_K)),)

    # Define strides
    strides = {
        'X': (SEQ_LEN * HEAD * HEAD_DIM, HEAD * HEAD_DIM, HEAD_DIM, 1),
        'COS': (HEAD_DIM, 1),
        'SIN': (HEAD_DIM, 1)
    }

    # Launch the kernel
    rotary_kernel[grid](
        X, COS, SIN, IS_VARLEN, BLOCK_M, BLOCK_K, INTERLEAVED, CONJUGATE, BATCH, HEAD, SEQ_LEN, HEAD_DIM
    )

    return X
