import triton
import triton.language as tl

# Define the BLOCK_SIZE
BLOCK_SIZE = 256

# Define the element-wise addition kernel
@triton.jit
def add_kernel(
    in_ptr0: tl.tensor,  # Pointer to the first input tensor
    in_ptr1: tl.tensor,  # Pointer to the second input tensor
    out_ptr: tl.tensor,  # Pointer to the output tensor
    n_elements: tl.int32,  # Number of elements to process
    BLOCK_SIZE: tl.constexpr,  # Block size
):
    # Get the program ID
    pid = tl.program_id(axis=0)
    # Calculate the starting index for the current block
    block_start = pid * BLOCK_SIZE
    # Create a range of offsets within the block
    offsets = tl.arange(0, BLOCK_SIZE)
    # Create a mask to prevent out-of-bound access
    mask = offsets < n_elements
    # Load values from input tensors using the mask
    x = tl.load(in_ptr0 + block_start + offsets, mask=mask)
    y = tl.load(in_ptr1 + block_start + offsets, mask=mask)
    # Perform element-wise addition
    z = x + y
    # Store the result back to the output tensor using the mask
    tl.store(out_ptr + block_start + offsets, z, mask=mask)

# Define the wrapper function for the kernel
def add_wrapper(x, y):
    # Create an output tensor with the same shape and dtype as the input tensors
    out = torch.zeros_like(x)
    # Get the total number of elements to process
    n_elements = x.numel()
    # Calculate the number of blocks needed
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the kernel
    add_kernel[(num_blocks,)](x, y, out, n_elements, BLOCK_SIZE)
    # Return the result
    return out
