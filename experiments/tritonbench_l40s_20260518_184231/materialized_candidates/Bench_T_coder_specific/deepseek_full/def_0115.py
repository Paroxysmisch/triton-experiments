import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def tanh_linear(input, weight, bias=None) -> Tensor:
    # Linear transformation followed by Tanh activation
    output = tl.dot(input, weight)
    if bias is not None:
        output += bias
    output = tl.tanh(output)
    return output

def test_tanh_linear():
    inp = torch.randn(2, 3, 4, 5)
    weight = torch.randn(7, 2)
    bias = torch.randn(7)
    triton_output = tanh_linear(inp, weight, bias)
    assert triton_output.shape == (2, 3, 4, 5, 7)
    torch_output = torch.tanh(torch.einsum('n...k,kd->n...d', inp, weight) + bias)
    assert torch_output.shape == (2, 3, 4, 5, 7)
    assert torch.allclose(triton_output, torch_output)

if __name__ == "__main__":
    test_tanh_linear()
