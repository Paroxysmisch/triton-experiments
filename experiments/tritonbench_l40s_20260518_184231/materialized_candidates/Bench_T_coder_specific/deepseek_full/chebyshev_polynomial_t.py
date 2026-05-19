import triton
import triton.language as tl
import torch

# Triton kernel for Chebyshev polynomial of the first kind T_n(input)
@triton.jit
def chebyshev_polynomial_t(input, n, **out):
    # If n = 0, return 1
    if n == 0:
        return 1
    # If n = 1, return input
    elif n == 1:
        return input
    # If n < 6 or |input| > 1, use recursive formula
    elif n < 6 or abs(input) > 1:
        T_n_minus_1 = chebyshev_polynomial_t(input, n - 1)
        T_n = chebyshev_polynomial_t(input, n)
        return 2 * input * T_n - T_n_minus_1
    # Otherwise, use trigonometric formula
    else:
        return tl.cos(n * tl.acos(input))

# Test function for Chebyshev polynomial
def test_chebyshev_polynomial_t():
    # Test data
    input_data = torch.tensor([0.5], dtype=torch.float32)
    n_data = torch.tensor([3], dtype=torch.float32)
    # Expected output
    expected_output = torch.tensor([0.75, 0.5, 0.25, 0.5, 0.75], dtype=torch.float32)
    # Call the Triton kernel
    output = chebyshev_polynomial_t(input_data, n_data)
    # Check if the output is close to the expected output
    assert torch.allclose(output, expected_output)
