import torch
import triton
import triton.language as tl

@triton.jit
def _quantize_global(
    x_ptr,
    absmax_inv_ptr,
    output_ptr,
    n_elements,
):
    # Calculate the program id and element offsets
    pid = tl.program_id(axis=0)
    offsets = pid * n_elements
    # Load the input and absolute max value
    x = tl.load(x_ptr + offsets)
    absmax_inv = tl.load(absmax_inv_ptr + pid)
    # Quantize the input and store the result
    output = tl.llrint(tl.scalar_t(127.0) * x * absmax_inv)
    tl.store(output_ptr + offsets, output)

def quantize_global(x: torch.Tensor):
    # Calculate the maximum absolute value and its inverse
    absmax = torch.max(torch.abs(x))
    absmax_inv = absmax ** -1

    # Prepare the output tensor
    output = torch.empty(*x.shape, device=x.device, dtype=torch.int8)

    # Ensure CUDA compatibility
    assert x.is_cuda and output.is_cuda

    # Define grid configuration
    grid = lambda meta: (x.shape[0],)

    # Launch the Triton kernel
    _quantize_global[grid](x, absmax_inv, output, n_elements=x.numel())

    return output, absmax
