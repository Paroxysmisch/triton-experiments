import triton
import triton.language as tl
import torch

@triton.jit
def pow_kernel(output_ptr, input_ptr, exponent_ptr, n_elements, is_scalar, BLOCK_SIZE: tl.constexpr):
    # Calculate the global index of the current thread
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask to prevent out-of-bounds access
    mask = idx < n_elements
    # Load input and exponent values
    x = tl.load(input_ptr + idx, mask=mask)
    if is_scalar:
        exponent = tl.load(exponent_ptr)  # Load the scalar exponent
    else:
        exponent = tl.load(exponent_ptr + idx, mask=mask)  # Load the tensor exponent
    # Compute the power
    result = tl.pow(x, exponent)
    # Store the result
    tl.store(output_ptr + idx, result, mask=mask)

def pow(input: torch.Tensor, exponent, *, out=None) -> torch.Tensor:
    # Ensure input is on CUDA
    assert input.is_cuda, "Input tensor must be on CUDA"
    # Determine if the exponent is a scalar or tensor
    is_scalar = isinstance(exponent, (int, float))
    # Prepare exponent tensor
    if is_scalar:
        exponent_tensor = torch.tensor(exponent, device=input.device, dtype=input.dtype)
    else:
        assert exponent.is_cuda, "Exponent tensor must be on CUDA"
        exponent_tensor = exponent
        # Check broadcastability
        assert torch.broadcast_shapes(input.shape, exponent.shape) == input.shape, \
            "Shapes of input and exponent must be broadcastable"

    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_cuda, "Output tensor must be on CUDA"
        assert out.shape == input.shape, "Output tensor must have the same shape as input"

    # Get the number of elements in the input
    n_elements = input.numel()
    # Determine block size for Triton kernel
    BLOCK_SIZE = 1024  # You can adjust this based on your GPU capabilities
    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    pow_kernel[grid](out, input, exponent_tensor, n_elements, is_scalar, BLOCK_SIZE=BLOCK_SIZE)

    return out
