import triton
import torch

@torch.jit.script
def relu_sqrt(input, inplace=False, out=None) -> torch.Tensor:
    if inplace:
        if out is None:
            out = input
        else:
            assert out is input, "Output must be the same as input for inplace operation"
    else:
        if out is None:
            out = input.clone()
        else:
            assert out.shape == input.shape, "Output shape must match input shape"

    # Launch the Triton kernel
    grid_size = (len(input) + 1023) // 1024
    relu_sqrt_kernel[(grid_size,), (1024,)](input.data_ptr(), out.data_ptr(), len(input))

    return out
