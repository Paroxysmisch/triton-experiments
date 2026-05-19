import torch
import triton
import triton.language as tl

def cos(input, *, out=None):
    assert isinstance(input, torch.Tensor), "Input must be a PyTorch tensor"
    
    if out is None:
        out = torch.empty_like(input)
    else:
        assert isinstance(out, torch.Tensor), "Output must be a PyTorch tensor"
        assert out.shape == input.shape, "Output shape must match input shape"

    n_elements = input.numel()
    grid_size = (n_elements + 255) // 256  # Assuming BLOCK_SIZE = 256

    cos_kernel[grid_size, 256](input.data_ptr(), out.data_ptr(), n_elements)

    return out
