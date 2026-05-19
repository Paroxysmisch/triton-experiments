import torch
import triton
import triton.language as tl

@triton.jit
def chebyshev_kernel(input_ptr, n, output_ptr, size):
    # Get the program index
    pid = tl.program_id(0)
    # Load input value
    x = tl.load(input_ptr + pid)
    # Initialize T0 and T1
    T0 = 1.0
    T1 = x
    # Determine the result based on the value of n
    if n == 0:
        result = T0
    elif n == 1:
        result = T1
    else:
        if n < 6 or abs(x) > 1:
            # Use the recursive formula
            for i in range(2, n + 1):
                T2 = 2 * x * T1 - T0
                T0, T1 = T1, T2
            result = T1
        else:
            # Use the trigonometric formula
            result = torch.cos(n * torch.acos(x))
    # Store the result
    tl.store(output_ptr + pid, result)

def chebyshev_polynomial_t(input: torch.Tensor, n: int, *, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    # Launch the Triton kernel
    chebyshev_kernel[(input.numel(),)](input, n, out, input.numel())
    return out

# Example usage
input_tensor = torch.tensor([0.5, 0.7, -0.3], dtype=torch.float32)
n = 4
output_tensor = chebyshev_polynomial_t(input_tensor, n)
print(output_tensor)
