import triton
import triton.language as tl

@triton.jit
def log_tanh_kernel(input_ptr, output_ptr, n_elements):
    # Get the index of the current element
    idx = tl.program_id(0) * tl.block_size(0) + tl.arange(0, tl.block_size(0))
    mask = idx < n_elements  # Ensure we don't go out of bounds

    # Load input tensor
    input_data = tl.load(input_ptr + idx)

    # Compute natural logarithm and then tanh
    output_data = tl.tanh(tl.log(input_data))

    # Store the result in the output tensor
    tl.store(output_ptr + idx, output_data, mask=mask)

def log_tanh(input: Tensor, out: Optional[Tensor] = None) -> Tensor:
    # Ensure all input elements are positive
    if not (input > 0).all():
        raise ValueError("All input elements must be positive for the logarithm function to be defined.")

    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(input)

    # Launch the Triton kernel
    grid = (input.numel() + 255) // 256  # Assuming a block size of 256
    log_tanh_kernel[grid](input, out, input.numel())

    return out
