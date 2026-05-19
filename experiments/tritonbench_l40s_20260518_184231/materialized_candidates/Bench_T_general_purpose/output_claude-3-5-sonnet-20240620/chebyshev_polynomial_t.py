import triton
import triton.language as tl

@triton.jit
def chebyshev_kernel(input_ptr, n_ptr, out_ptr, n_elements):
    # Load input and n values
    input_val = tl.load(input_ptr)
    n = tl.load(n_ptr)

    # Initialize output
    if n == 0:
        tl.store(out_ptr, 1.0)
    elif n == 1:
        tl.store(out_ptr, input_val)
    else:
        # Recursive calculation for n < 6 or |input| > 1
        if n < 6 or abs(input_val) > 1:
            T_n_minus_1 = 1.0  # T_0
            T_n = input_val    # T_1
            for i in range(2, n + 1):
                T_n_plus_1 = 2 * input_val * T_n - T_n_minus_1
                T_n_minus_1 = T_n
                T_n = T_n_plus_1
            tl.store(out_ptr, T_n)
        else:
            # Trigonometric formula for n >= 6 and |input| <= 1
            theta = tl.arccos(input_val)
            result = tl.cos(n * theta)
            tl.store(out_ptr, result)

# Wrapper function
def chebyshev_polynomial_t(input: tl.Tensor, n: tl.Tensor, *, out: tl.Tensor = None) -> tl.Tensor:
    if out is None:
        out = tl.empty_like(input)
    
    # Launch the kernel
    grid = (input.shape[0],)
    chebyshev_kernel[grid](input, n, out)
    return out
