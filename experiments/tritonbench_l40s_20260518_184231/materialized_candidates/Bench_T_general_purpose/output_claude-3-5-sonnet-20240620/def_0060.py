import triton
import triton.language as tl

@triton.jit
def exp_sqrt_kernel(input_ptr, output_ptr, n_elements):
    # Get the index of the current element
    idx = tl.program_id(0) * tl.block_size(0) + tl.arange(0, tl.block_size(0))
    mask = idx < n_elements  # Mask to avoid out-of-bounds access

    # Load input tensor
    input_val = tl.load(input_ptr + idx, mask=mask)

    # Compute exp and then sqrt
    exp_val = tl.exp(input_val)
    result = tl.sqrt(exp_val)

    # Store the result in the output tensor
    tl.store(output_ptr + idx, result, mask=mask)

def exp_sqrt(input: Tensor, out: Optional[Tensor] = None) -> Tensor:
    # Check if output tensor is provided, otherwise create a new one
    if out is None:
        out = torch.empty_like(input)

    # Get the number of elements in the input tensor
    n_elements = input.numel()

    # Launch the Triton kernel
    grid = (triton.cdiv(n_elements, 1024),)  # Assuming a block size of 1024
    exp_sqrt_kernel[grid](input, out, n_elements)

    return out
