import triton
import torch

@triton.jit
def tensordot_rsqrt(a, b, dims):
    # Perform tensor contraction
    tensor_contract = torch.tensordot(a, b, dims)

    # Apply reciprocal square root
    output = torch.rsqrt(tensor_contract)

    return output
