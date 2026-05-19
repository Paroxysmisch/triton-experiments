import torch
import triton
import triton.language as tl
from xformers.components import Activation
from xformers.triton.ops.softmax import _log_softmax, _softmax
from xformers.triton.ops.utils import warps_kernel_configs


@triton.autotune(
    configs=warps_kernel_configs(),
    key=["N", "HAS_LOG"],
)
@triton.heuristics({"HAS_DROPOUT": lambda args: args["p"] != 1.0})
@triton.heuristics({"HAS_INPLACE_SOFTMAX": lambda args: args["inplace"] and args["dim"] == -1})
@triton.jit
def fused_mixed_precision_softmax(x_ptr, y_ptr, M, N, stride_x_batch, stride_x_n,
                                  p: tl.constexpr, dim: tl.constexpr, inplace: tl.constexpr,
                                  block_size: tl.constexpr, HAS_BATCH_DIM: tl.constexpr,
                                  HAS_LOG: tl.constexpr, HAS_DROPOUT: tl.constexpr,
                                  HAS_INPLACE_SOFTMAX: tl.constexpr,
                                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Map program ids `pid_m` and `pid_n` to the part of the input they should compute.
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    if HAS_BATCH_DIM:
        batch_stride = BLOCK_M * BLOCK_N * stride_x_batch
        x_ptr += pid_b * batch_stride
    else:
        pid_b = 0
    # Compute the block that each program will go through
    # e.g. if there are 1024 rows and 128 threads we want exactly 8 blocks of 128 rows.
    # So we do (1024+127)//128 = 8
    # But if there are 1024 rows and only 64 threads we still want exactly 8 blocks of 128 rows each.
    # So we do (1024+127)//128 = 8 even though we could fit two 128 row blocks in one 256 row block.
    # However, if there are 913 rows and 64 threads we should only iterate 9 times since 913//64 = 14 but we can't fit another block.
    # So we do (913+64-1)//64 = 15 but we want (913+63)//64 = 14
    # When BLOCK_M is 64, (913+64-1)//64 = 15 but we want (913+63)//64 = 14
    # When BLOCK_M is 63, (913+64-1)//64 = 15 but we want (913+63)//64 = 15
    # We add BLOCK_M-1 before doing integer division so when BLOCK_M is a power-of-two we get exact results
    # e.g. when BLOCK_M is 64: (913+64-1)//64 = (913+63)//64 = 15
    # e.g. when BLOCK_M is 128: (913+128-1)//128 = (913+127)//128 = 8
    # See https://stackoverflow.com/questions/1487290/how-to-calculate-ceildiv-in-python
    # Note that C++ also uses this trick to perform ceil div https://godbolt.org/z/YKfWxqoY3
    # However, we can't use C++ style integer division because Triton doesn't support it
    # Using Python style // works just fine
    last_block = ((pid_m + 1) * BLOCK_M + block_size - 1) // block_size * block_size
    num_pid_m = ((N + BLOCK_N - 1) // BLOCK_N) * ((last_block + BLOCK_M - 1) // BLOCK_M)
    pid_m = (pid_m + pid_b * num_pid_m)
    block_start = pid_m * BLOCK_M
    offset_m = block_start + tl.arange(0, BLOCK_M)
    offset_n = tl.arange(0, BLOCK_N)

    # Create a mask to guard memory operations against out-of-bounds accesses
    mask = offset_m < M and offset_n < N

    # Load x into SRAM, using a regular unrolled load for simplicity, could be optimized further
    x = tl.load(x_ptr + offset_m[:, None] * stride_x_n + offset_n[None, :],
                mask=mask,
                other=-float("inf"))

    # Perform the fused operation
    if inplace:
        assert HAS_INPLACE_SOFTMAX
        # Use a scratch space in DRAM to save the intermediate values
        # We cannot reuse the input-output `x` buffer since the dropout may clobber it
        z_ptr = y_ptr + offset_m[:, None] * stride_x_n + offset_n[None, :]
        # Step 1: Matrix multiply
        z = tl.dot(x, vec, allow_tf32=True)
        tl.store(z_ptr, z, mask=mask)
        # Step 2: Log_softmax
        _log_softmax(z_ptr, dim, False, inplace, False, stride_x_batch, stride_x_n,
                     block_size, True, False, False, BLOCK_M, BLOCK_N)
    else:
        # Step 1: Matrix multiply
        z = tl.dot(x, vec, allow_tf32=True)
        # Step 2: Log_softmax
        if HAS_DROPOUT:
            if HAS_LOG:
                y = _log_softmax(z, dim, False, inplace, True, stride_x_batch, stride_x_n,
                                 block_size, False, False, False, BLOCK_M, BLOCK_N)
            else:
                y = _softmax(z, dim, inplace, stride_x_batch, stride_x_n,
                             block_size, False, False, False, BLOCK_M, BLOCK_N)
            # Step 3: Dropout
            keep_prob = 1.0 - p
            random_tensor = tl.rand(y.shape, dtype=y.dtype)
            dropout_mask = random_tensor > keep_prob
            if not inplace:
                y *= dropout_mask.to(dtype=y.dtype) / keep_prob
        else:
            y = _log_softmax(z, dim, HAS_LOG, inplace, False, stride_x_batch, stride_x_n,
                             block_size, False, False, False, BLOCK_M, BLOCK_N)

        # Write output to DRAM
        tl.store(y_ptr + offset_m[:, None] * stride_x_n + offset_n[None, :], y, mask=mask)


def fused_mv_logsoftmax_dropout(input, vec, p=0.5, training=True, inplace=False, dim=-1, *, out=None):
    """
    Fused kernel for matrix-vector multiplication, log-softmax activation, and dropout.

    Args:
        input (Tensor): Input tensor.
            Expected to be of shape [M, N] or [batch, M, N].
        vec (Tensor): Vector to be used in matrix-vector multiplication.
            Expected to be of shape [N].
        p (float): Probability for dropout.
            Default: 0.5
        training (bool): Indicate whether the operation is performed during training.
            Default: True
        inplace (bool): Indicate whether to perform inplace softmax.
            Default: False
        dim (int): Dimension along which softmax is computed.
            Default: -1
        out (Tensor, optional): Output tensor.

    Returns:
        Tensor: Resultant tensor after fused operation.
    """
    assert dim in [-1, input.ndim - 1], "Only dim=-1 is supported"

    input = input.contiguous()
    vec = vec.contiguous()

    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_contiguous()

    if input.dtype == torch.float16:
        block_size = 64
    else:
        block_size = 128

    M = input.numel() // input.size(-1)
    N = vec.size(0)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    with torch.cuda.device(input.device.index):
        fused_mixed_precision_softmax[grid](input, out, M, N,
                                            input.stride(0), input.stride(1),
                                            1.0 - p, dim, inplace,
                                            block_size,
                                            HAS_BATCH_DIM=input.ndim > 2,
                                            HAS_LOG=inplace,
                                            HAS_DROPOUT=training and p != 1.0,
                                            BLOCK_M=triton.next_power_of_2(block_size),
                                            BLOCK_N=triton.next_power_of_2(16)):
    return out
