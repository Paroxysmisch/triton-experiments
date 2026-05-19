import triton
import triton.language as tl
import torch

@triton.jit
def _fused_mv_logsoftmax_dropout_kernel(
    MATRIX, VEC, OUTPUT, stride_m, stride_v, stride_o,
    n, m, p, training, seed, dim: tl.constexpr
):
    pid = tl.program_id(0)
    # Each program handles one row of the matrix
    row = pid
    if row >= n:
        return

    # 1) Matrix-Vector multiplication: z = A[row] * v
    # Accumulate dot-product
    dot_val = tl.float32(0.)
    row_offset = row * stride_m
    for col in range(m):
        a_val = tl.load(MATRIX + row_offset + col)
        v_val = tl.load(VEC + col * stride_v)
        dot_val += a_val * v_val

    # 2) Log-Softmax along the specified dim
    # Here, z is effectively a single scalar for each row if dim in [0, -1].
    # We collectively need to handle all rows to find max and sum of exp,
    # but for demonstration, since log-softmax is row-wise for a 1D result,
    # we can store partial results for each row, then apply a single pass.
    # This kernel computes just the dot_val; the actual log-softmax is done
    # in a second pass or we approximate by computing it row by row.
    # We first store dot_val into output for partial use.
    tl.store(OUTPUT + row * stride_o, dot_val)

@triton.jit
def _fused_logsoftmax_dropout_postprocess_kernel(
    OUTPUT, stride_o, n, p, training, seed
):
    pid = tl.program_id(0)
    row = pid
    if row >= n:
        return

    # 1) Load the dot value
    offset = row * stride_o
    val = tl.load(OUTPUT + offset)

    # 2) Subtract max (collected from entire set) to improve numerical stability
    # We'll do a parallel pass, so we replicate the logic:
    #    val = val - maxVal
    # We'll load maxVal from index 0. In real practice we'd store it in global memory.
    # For simplicity, store global maximum in OUTPUT[n] if the memory is allocated for it.
    max_val = tl.load(OUTPUT + n * stride_o)
    val = val - max_val

    # 3) Exponentiate and store back for the summation pass
    val = tl.exp(val)
    tl.store(OUTPUT + offset, val)

@triton.jit
def _fused_logsoftmax_dropout_sum_kernel(
    OUTPUT, stride_o, n, p, training, seed
):
    pid = tl.program_id(0)
    row = pid
    if row >= n:
        return

    # 1) In a typical softmax, we'd sum across rows after exponentiation.
    #    We do that here as a single pass with atomic additions or parallel reduce.
    # For demonstration, let's store the sum in OUTPUT[n+1].
    # Then we'll do a pass to finalize log-softmax per row.
    val = tl.load(OUTPUT + row * stride_o)
    # Atomic add to global location
    old = tl.atomic_add(OUTPUT + (n+1)*stride_o, val)

@triton.jit
def _fused_logsoftmax_dropout_finalize_kernel(
    OUTPUT, stride_o, n, p, training, seed
):
    pid = tl.program_id(0)
    row = pid
    if row >= n:
        return

    # Load exponentiated value
    offset = row * stride_o
    val = tl.load(OUTPUT + offset)
    # Load global sum
    sum_val = tl.load(OUTPUT + (1 + n) * stride_o)
    # Compute final log-softmax
    val = tl.log(val / sum_val)

    # 3) Dropout
    if training > 0:
        # Create a mask from linear congruential generator or just a simple approach
        # For demonstration, let's emulate:
        rng = tl.program_id(0) + seed  # not a good RNG, but a placeholder
        random_val = tl.rand(rng)      # advanced versions of Triton may not have this yet
        mask = random_val > p
        # Apply dropout
        val = tl.where(mask, val, 0.0)

    tl.store(OUTPUT + offset, val)

def fused_mv_logsoftmax_dropout(input, vec, p=0.5, training=True, inplace=False, dim=0, *, out=None):
    """
    Performs matrix-vector multiplication, then log-softmax, then dropout.

    Args:
        input (torch.Tensor): 2D tensor of shape [n, m].
        vec (torch.Tensor): 1D tensor of shape [m].
        p (float): Probability for dropout.
        training (bool): Whether in training mode or not.
        inplace (bool): Not used in this reference implementation.
        dim (int): Dimension along which log-softmax is computed.
        out (torch.Tensor): If provided, the result is written to it.

    Returns:
        torch.Tensor: Output tensor after fused operation.
    """
    assert input.dim() == 2, "input must be 2D"
    assert vec.dim() == 1, "vec must be 1D"
    n, m = input.shape
    assert vec.shape[0] == m, "vector size must match input's second dimension"

    # Allocate output
    if out is None:
        out = torch.empty_like(input[:, 0])  # shape [n]

    grid = (n,)
    # 1) Launch kernel to do matrix-vector multiplication, store in out
    _fused_mv_logsoftmax_dropout_kernel[grid](
        input, vec, out,
        input.stride(0), vec.stride(0), out.stride(0),
        n, m, p, int(training), 0, dim
    )

    # Compute maximum across the final dimension for log-softmax
    max_val = torch.max(out)
    # Write max_val to out[n] if we have enough space, else expand storage
    # We'll create a buffer to store out plus 2 more slots
    buffer = torch.empty(n+2, dtype=out.dtype, device=out.device)
    buffer[:n] = out
    buffer[n] = max_val
    buffer[n+1] = 0.0  # for sum

    # 2) Exponentiate partial result
    _fused_logsoftmax_dropout_postprocess_kernel[grid](
        buffer, buffer.stride(0), n, p, int(training), 0
    )

    # 3) Sum for softmax denominator
    _fused_logsoftmax_dropout_sum_kernel[grid](
        buffer, buffer.stride(0), n, p, int(training), 0
    )

    # 4) Finalize log-softmax + dropout
    _fused_logsoftmax_dropout_finalize_kernel[grid](
        buffer, buffer.stride(0), n, p, int(training), 0
    )

    # Copy results back
    out.copy_(buffer[:n])
    return out
