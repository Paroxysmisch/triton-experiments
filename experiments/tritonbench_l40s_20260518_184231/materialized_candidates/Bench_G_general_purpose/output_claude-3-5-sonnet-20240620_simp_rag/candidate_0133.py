import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(axis=0)
    
    # Compute the start index for this block
    block_start = pid * BLOCK_SIZE
    
    # Create a range of offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle the case where n_elements is not a multiple of BLOCK_SIZE
    mask = offsets < n_elements
    
    # Load the inputs
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform the addition
    output = x + y
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def add_wrapper(x, y, BLOCK_SIZE=1024):
    # Get the shape of the input tensors
    assert x.shape == y.shape, "Input tensors must have the same shape"
    n_elements = x.numel()
    
    # Compute the grid size
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create the output tensor
    output = torch.empty_like(x)
    
    # Launch the kernel
    add_kernel[(grid,)](
        x, y, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Compile the kernel and print the assembly
compiled_kernel = triton.compile(add_kernel, signature="*fp32,*fp32,*fp32,i32", constants={"BLOCK_SIZE": 1024})
print(compiled_kernel.asm["ttgir"])
