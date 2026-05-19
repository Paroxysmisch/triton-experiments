import torch

def fused_gather_masked_fill(input, dim, index, mask, value, sparse_grad=False, out=None):
    # Gather values from input tensor along specified dimension
    gathered = torch.gather(input, dim, index)

    # Replace gathered elements with specified value where mask is True
    masked_filled = torch.masked_fill(gathered, mask, value)

    return masked_filled
