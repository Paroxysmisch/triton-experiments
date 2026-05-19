import torch
import triton
import triton.language as tl

@triton.jit
def fused_gather_masked_fill(input, dim, index, mask, value, **META):
    # Define meta information for Triton kernel
    META["sparse_grad"] = False
    META["has_out"] = False
    META["has_mask"] = True
    META["has_index"] = True
    META["has_dim"] = True
    META["has_value"] = True
    META["has_input"] = True
    # Perform fused operation using torch.gather and torch.Tensor.masked_fill
    gathered = torch.gather(input, dim, index)
    return gathered.masked_fill(mask, value)

def fused_gather_masked_fill(input, dim, index, mask, value, *, sparse_grad=False, out=None):
    # Call the Triton kernel with provided arguments
    return fused_gather_masked_fill(input, dim, index, mask, value, sparse_grad=sparse_grad, out=out)
