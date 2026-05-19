import triton
import triton.language as tl

@triton.jit
def sub_gelu_kernel(input_ptr, other_ptr, alpha, out_ptr, n_elements, approximate):
    # Get the index of the current element
    idx = tl.program_id(0)
    
    # Ensure we do not exceed the number of elements
    if idx >= n_elements:
        return

    # Load input and other values
    input_val = tl.load(input_ptr + idx)
    other_val = tl.load(other_ptr + idx)

    # Perform the subtraction
    result = input_val - alpha * other_val

    # Apply GELU activation function
    if approximate == 'none':
        # Exact GELU
        gelu_val = result * tl.erf(result / tl.sqrt(2)) / 2 + result / 2
    elif approximate == 'tanh':
        # Approximate GELU using tanh
        gelu_val = 0.5 * result * (1 + tl.tanh(tl.sqrt(2 / tl.pi) * (result + 0.044715 * result ** 3)))
    else:
        raise ValueError("Invalid approximation method. Use 'none' or 'tanh'.")

    # Store the result in the output tensor
    tl.store(out_ptr + idx, gelu_val)

def sub_gelu(input: Tensor, other: Tensor or Number, alpha: float = 1, approximate: str = 'none', out: Tensor = None) -> Tensor:
    # Ensure input and other are tensors
    if isinstance(other, (int, float)):
        other = torch.full_like(input, other)

    # Prepare output tensor if not provided
    if out is None:
        out = torch.empty_like(input)

    # Get the number of elements
    n_elements = input.numel()

    # Launch the Triton kernel
    sub_gelu_kernel[(n_elements,)](input.data_ptr(), other.data_ptr(), alpha, out.data_ptr(), n_elements, approximate)

    return out
