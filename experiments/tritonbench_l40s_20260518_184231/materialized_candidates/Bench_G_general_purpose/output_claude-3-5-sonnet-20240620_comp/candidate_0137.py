import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    in_ptr0,  # pointer to first input tensor
    in_ptr1,  # pointer to second input tensor
    out_ptr,  # pointer to output tensor
    n_elements,  # number of elements in tensor
    BLOCK_SIZE: tl.constexpr,  # size of the block
):
    # compute unique program ID
    pid = tl.program_id(axis=0)
    
    # calculate starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # create mask to handle boundary conditions
    mask = offsets < n_elements
    
    # load data from first input tensor
    x = tl.load(in_ptr0 + offsets, mask=mask)
    # load data from second input tensor
    y = tl.load(in_ptr1 + offsets, mask=mask)
    
    # perform element-wise addition
    output = x + y
    
    # store result
    tl.store(out_ptr + offsets, output, mask=mask)

def add_wrapper(x: torch.Tensor, y: torch.Tensor):
    # ensure inputs are on GPU
    assert x.is_cuda and y.is_cuda
    # ensure inputs have same shape
    assert x.shape == y.shape
    
    # initialize output tensor
    out = torch.zeros_like(x)
    
    # define block size (can be tuned for performance)
    BLOCK_SIZE = 1024
    
    # calculate number of elements
    n_elements = x.numel()
    
    # calculate grid size
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # launch kernel
    add_kernel[(num_blocks,)](
        x.data_ptr(),
        y.data_ptr(),
        out.data_ptr(),
        n_elements,
        BLOCK_SIZE,
    )
    
    return out
