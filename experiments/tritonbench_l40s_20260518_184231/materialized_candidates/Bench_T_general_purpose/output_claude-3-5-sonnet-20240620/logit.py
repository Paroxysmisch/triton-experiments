import triton
import triton.language as tl

@triton.jit
def logit_kernel(input_ptr, output_ptr, eps, n_elements):
    # Get the index of the current element
    idx = tl.program_id(0)
    
    # Ensure we do not exceed the number of elements
    if idx >= n_elements:
        return

    # Load the input value
    x = tl.load(input_ptr + idx)

    # Handle the clamping based on eps
    if eps is not None:
        z = tl.where(x < eps, eps, tl.where(x > 1 - eps, 1 - eps, x))
    else:
        z = x

    # Compute the logit
    logit_value = tl.log(z / (1 - z)) if (z > 0) & (z < 1) else float('nan')

    # Store the result
    tl.store(output_ptr + idx, logit_value)

def logit(input: torch.Tensor, eps: Optional[float] = None, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Check if input is a valid tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    # Prepare output tensor if not provided
    if out is None:
        out = torch.empty_like(input)

    # Get the number of elements
    n_elements = input.numel()

    # Launch the Triton kernel
    logit_kernel[(n_elements,)](input, out, eps, n_elements)

    return out
