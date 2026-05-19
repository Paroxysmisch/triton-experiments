import torch
from torch.testing import assert_close
from torch._inductor.triton_heuristics import triton_helpers
import triton
import triton.language as tl
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@triton.jit
def triton_kernel_signbit(inp):
    # Triton kernel for signbit operation
    return tl.signbit(inp)

@triton.jit
def triton_kernel_bitwise_and(inp, other):
    # Triton kernel for bitwise_and operation
    return tl.bitwise_and(inp, other)

def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    # Function to call the Triton kernels
    assert other.is_integral or other.is_boolean, "Other tensor must be of integral or boolean types"
    signbit_result = triton_kernel_signbit(input)
    bitwise_and_result = triton_kernel_bitwise_and(input, other)
    return signbit_result, bitwise_and_result

def test_signbit_bitwise_and():
    # Test function for signbit_bitwise_and
    input = torch.tensor([0.7, -1.2, 0., 2.3], dtype=torch.float32)
    other = torch.tensor([1, 0, 1, 1], dtype=torch.int8)
    signbit_result_expected = torch.tensor([False, True, False, False])
    bitwise_and_result_expected = torch.tensor([0, 0, 0, 0], dtype=torch.int8)
    signbit_result, bitwise_and_result = signbit_bitwise_and(input, other)
    assert_close(signbit_result, signbit_result_expected)
    assert_close(bitwise_and_result, bitwise_and_result_expected)

if __name__ == "__main__":
    test_signbit_bitwise_and()
