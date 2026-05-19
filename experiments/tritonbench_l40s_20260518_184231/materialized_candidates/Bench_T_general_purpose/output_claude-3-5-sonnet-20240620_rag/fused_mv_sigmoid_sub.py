import triton
import triton.language as tl

@triton.jit
def fused_mv_sigmoid_sub(input, vec, other, alpha, out, n, m):
    """
    Performs a fused operation combining matrix-vector multiplication,
    sigmoid activation, and subtraction.

    Args:
        input: Input matrix A of shape (n, m).
        vec: Input vector v of shape (m).
        other: Tensor or scalar b to subtract from the sigmoid output, scaled by alpha.
        alpha: Scalar multiplier for other.
        out: Output tensor. Ignored if None.
        n: Number of rows in the input matrix.
        m: Number of columns in the input matrix.

    Returns:
        Tensor after performing the fused operation.
    """
    row = tl.program_id(0)
    if row >= n:
        return

    # Matrix-vector multiplication
    z = tl.dot(input[row, :], vec)

    # Sigmoid activation
    s = 1 / (1 + tl.exp(-z))

    # Subtraction
    if isinstance(other, (int, float)):
        out[row] = s - alpha * other
    else:
        out[row] = s - alpha * other[row]

@triton.jit
def wrapper_fused_mv_sigmoid_sub(input, vec, other, alpha=1, out=None):
    n, m = input.shape
    if out is None:
        out = tl.empty((n,), dtype=tl.float32)

    fused_mv_sigmoid_sub(input, vec, other, alpha, out, n, m)
    return out
