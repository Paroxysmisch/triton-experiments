import triton.language as tl
import torch

@triton.jit
def relu_kernel(input_ptr, output_ptr, N, block_start, offsets):
    # Compute the unique id for each program running in the grid
    pid = tl.program_id(axis=0)
    # Compute the unique id for each thread in the kernel
    tid = tl.thread_id(axis=0)
    # Compute the block size for this kernel instance
    block_size = tl.cdiv(N, tl.cumsum(offsets, axis=0))[pid]
    # Calculate the start index of the current block
    block_start = block_start[pid] 
    # Compute mask of items thread loads
    mask = tid < block_size 
    # Compute global index for each item in the block,
    # relative to the full tensor data size
    gindex = block_start + (tid * offsets[pid])
    # Only active threads perform the ReLU operation
    # For elements outside the range, the mask ensures they're preserved
    tmp = tl.load(input_ptr + gindex, mask=mask)
    output = tl.where(tmp > 0, tmp, 0)  # Apply ReLU
    tl.store(output_ptr + gindex, output, mask=mask)  # Store back to memory

def relu(x):
    N = x.numel()
    block_size = 1024
    num_warps = 4
    # Get the pointer to the data in CUDA memory
    input_ptr = torch.cuda.memory.torch_tensor_ptr(x)
    output = torch.empty_like(x)
    output_ptr = torch.cuda.memory.torch_tensor_ptr(output)
    # Build offsets array that allows us to calculate the start element for each group of threads in the kernel
    offsets = torch.arange(0, N, block_size * num_warps, device='cuda')
    # Calculate the number of blocks we need to process the matrix
    num_warps = tl.cdiv(N, block_size * num_warps)
    block_start = torch.arange(0, num_warps * block_size * num_warps, block_size * num_warps, device='cuda')
    # Run the kernel to perform ReLU operation
    relu_kernel[(num_warps,)](input_ptr, output_ptr, N, block_start, offsets)
    return output
