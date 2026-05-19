import triton
import triton.language as tl

@triton.jit
def pow_kernel(input_ptr, exponent_ptr, output_ptr, N, is_scalar):
    # Compute the index for each element
    idx = tl.program_id(0) * tl.block_size(0) + tl.arange(0, tl.block_size(0))
    mask = idx < N  # Mask to avoid out-of-bounds access

    # Load input and exponent
    x = tl.load(input_ptr + idx, mask=mask)
    if is_scalar:
        exponent = tl.load(exponent_ptr)  # Load scalar exponent
        result = x ** exponent
    else:
        exponent = tl.load(exponent_ptr + idx, mask=mask)  # Load tensor exponent
        result = x ** exponent

    # Store the result
    tl.store(output_ptr + idx, result, mask=mask)


def pow(input: torch.Tensor, exponent: Union[float, torch.Tensor], *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Check if exponent is a tensor or scalar
    is_scalar = isinstance(exponent, float)
    
    # Prepare the output tensor if not provided
    if out is None:
        out = torch.empty_like(input)

    # Get the number of elements
    N = input.numel()

    # Launch the Triton kernel
    grid = (triton.cdiv(N, 1024),)  # Assuming a block size of 1024
    if is_scalar:
        triton.launch(pow_kernel, grid=grid, inputs=[input, torch.tensor(exponent), out, N, is_scalar])
    else:
        triton.launch(pow_kernel, grid=grid, inputs=[input, exponent, out, N, is_scalar])

    return out
