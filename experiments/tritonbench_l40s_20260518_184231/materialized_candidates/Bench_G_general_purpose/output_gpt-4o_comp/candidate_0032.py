import triton
import triton.language as tl
import torch

# Define the Triton kernel for dropout
@triton.jit
def _dropout(x_ptr, x_keep_ptr, output_ptr, n_elements, p, BLOCK_SIZE: tl.constexpr):
    # Program ID and offset calculation
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Offsets for loading data
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements

    # Load input data and mask
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask, other=0.0)

    # Apply dropout
    output = tl.where(x_keep, x / (1 - p), 0.0)

    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

# Define the host function for dropout
def dropout(x, x_keep, p):
    assert x.is_contiguous(), "Input tensor x must be contiguous"
    assert x_keep.is_contiguous(), "Mask tensor x_keep must be contiguous"
    assert x.shape == x_keep.shape, "Input tensor and mask tensor must have the same shape"

    # Number of elements
    n_elements = x.numel()

    # Define block size
    BLOCK_SIZE = 1024

    # Compute grid size
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Allocate output tensor
    output = torch.empty_like(x)

    # Launch the Triton kernel
    _dropout[grid_size](
        x_ptr=x.data_ptr(),
        x_keep_ptr=x_keep.data_ptr(),
        output_ptr=output.data_ptr(),
        n_elements=n_elements,
        p=p,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output

# Example usage
if __name__ == "__main__":
    # Define input tensor and mask
    x = torch.rand(2048, device='cuda')
    x_keep = (torch.rand(2048, device='cuda') < 0.8).float()  # 80% keep rate

    # Apply dropout
    output = dropout(x, x_keep, p=0.2)
    print(output)
