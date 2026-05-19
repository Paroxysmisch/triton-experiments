import triton
import triton.language as tl
import torch

@triton.jit
def _swiglu_fwd_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,  # number of elements in each tensor
    BLOCK_SIZE: tl.constexpr
):
    # Define the program ID and the starting index for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Create a range of indices for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask to ensure we do not read/write out of bounds
    mask = offsets < n_elements

    # Load x and y from memory
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute the Swiglu operation: x * sigmoid(x) * y
    sigmoid_x = 1 / (1 + tl.exp(-x))
    out = x * sigmoid_x * y

    # Store the result back to memory
    tl.store(out_ptr + offsets, out, mask=mask)

def _swiglu_fwd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensors are contiguous
    x = x.contiguous()
    y = y.contiguous()

    # Get the number of elements
    n_elements = x.numel()

    # Allocate output tensor
    out = torch.empty_like(x)

    # Define block size
    BLOCK_SIZE = 1024

    # Calculate the number of blocks
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    _swiglu_fwd_kernel[grid](
        x_ptr=x.data_ptr(),
        y_ptr=y.data_ptr(),
        out_ptr=out.data_ptr(),
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out

# Example usage:
x = torch.randn(2048, device='cuda')
y = torch.randn(2048, device='cuda')
out = _swiglu_fwd(x, y)
