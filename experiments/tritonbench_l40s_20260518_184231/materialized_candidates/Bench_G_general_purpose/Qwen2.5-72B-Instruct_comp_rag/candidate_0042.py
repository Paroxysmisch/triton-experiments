import triton
import triton.language as tl
import torch

@triton.jit
def pow_func_scalar_tensor_kernel_rank_1(
    X,  # Pointer to the input tensor
    Y,  # Pointer to the output tensor
    scalar,  # Scalar value for exponentiation
    N,  # Number of elements in the tensor
    BLOCK_SIZE: tl.constexpr  # Block size for parallel processing
):
    # Compute the starting index for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Compute the ending index for this block
    block_end = min(block_start + BLOCK_SIZE, N)

    # Load the data into a Triton block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < block_end
    x = tl.load(X + offsets, mask=mask)

    # Compute the power of each element
    y = tl.math.pow(x, scalar)

    # Store the result back to the output tensor
    tl.store(Y + offsets, y, mask=mask)

def pow_func_scalar_tensor_wrapper_rank_1(
    input_tensor: torch.Tensor,
    scalar: float,
    output_tensor: Optional[torch.Tensor] = None
):
    # Ensure input and output tensors are on the same device
    device = input_tensor.device
    if output_tensor is None:
        output_tensor = torch.empty_like(input_tensor, device=device)

    # Ensure input and output tensors have the same shape
    assert input_tensor.shape == output_tensor.shape, "Input and output tensors must have the same shape"

    # Get the number of elements in the tensor
    N = input_tensor.numel()

    # Determine the optimal block size
    BLOCK_SIZE = 1024  # Example block size, can be tuned for better performance

    # Determine the grid size
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    # Launch the kernel
    pow_func_scalar_tensor_kernel_rank_1[grid](
        input_tensor,  # Pointer to the input tensor
        output_tensor,  # Pointer to the output tensor
        scalar,  # Scalar value for exponentiation
        N,  # Number of elements in the tensor
        BLOCK_SIZE  # Block size for parallel processing
    )

    return output_tensor
