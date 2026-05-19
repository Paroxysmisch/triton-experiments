import triton
import triton.language as tl

@triton.jit
def polygamma_kernel(n, input_ptr, output_ptr, n_elements):
    # Compute the n-th derivative of the digamma function
    # using the input tensor and store the result in the output tensor.
    idx = tl.program_id(0)
    if idx >= n_elements:
        return

    x = tl.load(input_ptr + idx)
    result = 0.0

    # Compute the n-th derivative of the digamma function
    for k in range(n + 1):
        result += (-1)**k * tl.math.gamma(k + 1) * tl.math.pow(x, -k - 1)

    tl.store(output_ptr + idx, result)

def polygamma(n: int, input: Tensor, *, out: Tensor = None) -> Tensor:
    # Ensure n is a nonnegative integer
    if n < 0:
        raise ValueError("n must be a nonnegative integer.")
    
    # Prepare output tensor if not provided
    if out is None:
        out = torch.empty_like(input)

    # Launch the Triton kernel
    grid = (input.numel(),)
    polygamma_kernel[grid](n, input.data_ptr(), out.data_ptr(), input.numel())
    
    return out
