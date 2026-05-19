import torch
import triton
import triton.language as tl
from triton.language.libdevice import div_rn

@triton.jit
def reciprocal(input):
    # Convert integral inputs to the default scalar type
    input = input.to(tl.float32)
    # Calculate the reciprocal of the elements
    output = div_rn(1.0, input)
    return output

def test_reciprocal():
    """Test function for reciprocal"""
    input = torch.tensor([1.0, 2.0, 4.0], device='cuda')
    output = torch.empty_like(input)
    reciprocal(input, out=output)
    expected = torch.reciprocal(input)
    assert torch.allclose(output, expected)
    print("Test passed")

if __name__ == "__main__":
    test_reciprocal()
