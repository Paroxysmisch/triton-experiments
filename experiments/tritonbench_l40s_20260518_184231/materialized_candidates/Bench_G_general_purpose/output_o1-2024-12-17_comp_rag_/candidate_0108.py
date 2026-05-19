import triton
import triton.language as tl


@triton.jit
def rmsnorm_triton(
    x_ptr: tl.pointer_type,       # x: float32[batch, M, K]
    rms_w_ptr: tl.pointer_type,   # rms_weights: float32[K]
    out_ptr: tl.pointer_type,     # out: float32[batch, M, K]
    stride_b: tl.int32,           # stride along the batch dimension in x/out
    stride_m: tl.int32,           # stride along the M dimension in x/out
    stride_k: tl.int32,           # stride along the K dimension in x/out
    B: tl.int32,                  # batch size
    M: tl.int32,                  # dimension M
    K: tl.int32,                  # dimension K
    eps: tl.float32,              # small constant for numerical stability
    BLOCK_N_SIZE: tl.constexpr,   # number of threads/chunk size along K
):
    # Identify which (batch, M) this program instance should process
    b = tl.program_id(0)
    m = tl.program_id(1)
    # Check bounds (though assuming kernel launch grid = (B, M))
    if b >= B or m >= M:
        return

    # Compute the starting offset for this row in x/out
    row_offset = b * stride_b + m * stride_m

    # Pass 1: compute sum of squares along the K dimension in chunks
    sum_squares = tl.float32(0.)
    k_iter = 0
    while k_iter < K:
        offsets = k_iter + tl.arange(0, BLOCK_N_SIZE)
        mask = offsets < K
        x_vals = tl.load(
            x_ptr + row_offset + offsets * stride_k,
            mask=mask,
            other=0.0
        )
        sum_squares += tl.sum(x_vals * x_vals, mask=mask)
        k_iter += BLOCK_N_SIZE

    # Compute RMS
    rms = tl.sqrt(sum_squares / tl.float32(K) + eps)

    # Pass 2: normalize and store
    k_iter = 0
    while k_iter < K:
        offsets = k_iter + tl.arange(0, BLOCK_N_SIZE)
        mask = offsets < K
        x_vals = tl.load(
            x_ptr + row_offset + offsets * stride_k,
            mask=mask,
            other=0.0
        )
        w_vals = tl.load(rms_w_ptr + offsets, mask=mask, other=1.0)
        out_vals = (x_vals / rms) * w_vals
        tl.store(
            out_ptr + row_offset + offsets * stride_k,
            out_vals,
            mask=mask
        )
        k_iter += BLOCK_N_SIZE


def rmsnorm_wrapper(x, rms_w, out, eps, BLOCK_N_SIZE, num_warps=4):
    """
    x, rms_w, out are torch Tensors on GPU with shapes:
      x:      [batch, M, K]
      rms
