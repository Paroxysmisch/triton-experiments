import triton
import triton.language as tl
import torch

# Approximate digamma using a common approach:
# 1) For x <= 0 or close to 0, we set the result to -inf when x == 0 (as per PyTorch >= 1.8).
#    For negative or non-positive values (other than 0), this simple example does not implement
#    reflection or error checks, and uses -inf as a placeholder.
# 2) For smaller positive x, repeatedly shift upward using the identity:
#      digamma(x) = digamma(x+1) - 1/x
#    until x >= SHIFT_LIMIT, or until the value is large enough for a series approximation.
# 3) For large x, apply an asymptotic series expansion:
#      digamma(x) ~ ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - ...

@triton.jit
def _digamma_kernel(
    in_ptr, out_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
    SHIFT_LIMIT: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(in_ptr + offsets, mask=mask)

    # Initialize result
    result = tl.zeros_like(x)

    # Handle x == 0 => -Inf
    is_zero = x == 0.0
    # Mark negative or <= 0 with same placeholder -Inf (simple example)
    is_neg = x < 0.0
    # To avoid division by zero in the shift loop
    x = tl.where(is_zero | is_neg, 1.0, x)

    # Shift up for smaller x
    shift_sum = tl.zeros_like(x)
    cond_small = x < SHIFT_LIMIT
    while tl.any(cond_small):
        x_new = tl.where(cond_small, x + 1.0, x)
        shift_sum_new = tl.where(cond_small, shift_sum - 1.0 / x, shift_sum)
        x = x_new
        shift_sum = shift_sum_new
        cond_small = x < SHIFT_LIMIT

    # Asymptotic expansion for large x
    # digamma(x) ~ ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) ...
    inv_x = 1.0 / x
    inv_x2 = inv_x * inv_x
    term1 = tl.log(x)
    term2 = -0.5 * inv_x
    term3 = -1.0 / 12.0 * inv_x2
    term4 = 1.0 / 120.0 * inv_x2 * inv_x2  # 1/x^4
    term5 = -1.0 / 252.0 * inv_x2 * inv_x2 * inv_x2  # 1/x^6
    # Sum series terms
    approx_large = term1 + term2 + term3 + term4 + term5

    # Combine shift + large-x approximation
    result = approx_large + shift_sum

    # Apply special cases
    result = tl.where(is_zero, float('-inf'), result)
    result = tl.where(is_neg, float('-inf'), result)

    tl.store(out_ptr + offsets, result, mask=mask)


def digamma(input, *, out=None):
    """
    digamma(input, *, out=None) -> Tensor

    Computes the digamma function (the logarithmic derivative of the gamma function)
    on the input tensor. If out is specified, the result is placed in that tensor.

    From PyTorch 1.8 onwards, digamma(0) returns -Inf (instead of NaN in earlier versions).
    """
    if out is None:
        out = torch.empty_like(input)

    # Flatten input/output to 1D for Triton kernel
    input_flat = input.contiguous().view(-1)
    out_flat = out.contiguous().view(-1)

    n = input_flat.numel()
    if n == 0:
        return out

    # Launch kernel
    BLOCK_SIZE = 1024
    grid = ( (n + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    SHIFT_LIMIT = 5.0

    _digamma_kernel[grid](
        in_ptr=input_flat, 
        out_ptr=out_flat,
        n=n,
        BLOCK_SIZE=BLOCK_SIZE,
        SHIFT_LIMIT=SHIFT_LIMIT,
    )

    return out.reshape(input.shape)
