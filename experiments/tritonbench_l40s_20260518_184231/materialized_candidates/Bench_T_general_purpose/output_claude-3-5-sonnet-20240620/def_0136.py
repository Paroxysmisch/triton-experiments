import triton
import triton.language as tl

@triton.jit
def softmax_kernel(input_ptr, output_ptr, dim, N, BLOCK_SIZE: tl.constexpr):
    # Calculate the index of the current element
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK_SIZE)

    # Load input data
    input_data = tl.load(input_ptr + row * N + col)

    # Compute max for numerical stability
    max_val = tl.max(input_data)

    # Compute exp(x - max) for softmax
    exp_data = tl.exp(input_data - max_val)

    # Compute sum of exp_data
    sum_exp = tl.sum(exp_data)

    # Compute softmax
    softmax_output = exp_data / sum_exp

    # Store the result
    tl.store(output_ptr + row * N + col, softmax_output)

def softmax(input: Tensor, dim: int, dtype=None) -> Tensor:
    # Ensure input is a tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    # Cast input to the desired dtype if specified
    if dtype is not None:
        input = input.to(dtype)

    # Get the shape of the input tensor
    shape = input.shape
    N = shape[dim]  # Size along the specified dimension

    # Create output tensor
    output = torch.empty_like(input)

    # Launch the Triton kernel
    grid = (shape[0],)  # Assuming the first dimension is the batch size
    softmax_kernel[grid](input, output, dim, N, BLOCK_SIZE=1024)

    return output
