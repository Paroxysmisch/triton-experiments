import triton
import triton.language as tl
import torch

@triton.jit
def var_len_copy_kernel_triton(
    # Pointers to input and output arrays
    src_ptr,           # Source array pointer
    dst_ptr,           # Destination array pointer
    src_start_ptr,     # Source segment start indices pointer
    dst_start_ptr,     # Destination segment start indices pointer
    lengths_ptr,       # Segment lengths pointer
    num_segments,      # Total number of segments
    BLOCK_SIZE: tl.constexpr,  # Static block size for optimization
):
    # Get program ID for this thread
    pid = tl.program_id(0)
    
    # Each block handles one segment
    if pid >= num_segments:
        return
        
    # Load segment information
    src_start = tl.load(src_start_ptr + pid)
    dst_start = tl.load(dst_start_ptr + pid)
    length = tl.load(lengths_ptr + pid)
    
    # Copy elements in blocks of BLOCK_SIZE
    for i in range(0, length, BLOCK_SIZE):
        # Calculate number of elements to copy in this block
        block_length = tl.minimum(BLOCK_SIZE, length - i)
        
        # Generate offsets for this block
        offs = tl.arange(0, block_length)
        
        # Load source elements
        src_block = tl.load(src_ptr + src_start + i + offs,
                          mask=offs < block_length)
        
        # Store to destination
        tl.store(dst_ptr + dst_start + i + offs,
                src_block,
                mask=offs < block_length)

def launch_var_len_copy_triton(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_starts: torch.Tensor,
    dst_starts: torch.Tensor,
    lengths: torch.Tensor,
    BLOCK_SIZE: int = 128
):
    """
    Wrapper function to launch the Triton kernel
    
    Args:
        src: Source tensor
        dst: Destination tensor
        src_starts: Starting indices in source tensor
        dst_starts: Starting indices in destination tensor
        lengths: Lengths of segments to copy
        BLOCK_SIZE: Block size for processing (power of 2 recommended)
    """
    # Ensure inputs are on GPU and have correct dtype
    assert src.is_cuda and dst.is_cuda
    assert src_starts.is_cuda and dst_starts.is_cuda and lengths.is_cuda
    
    num_segments = lengths.shape[0]
    
    # Launch kernel with appropriate grid size
    grid = (num_segments,)
    
    var_len_copy_kernel_triton[grid](
        src_ptr=src,
        dst_ptr=dst,
        src_start_ptr=src_starts,
        dst_start_ptr=dst_starts,
        lengths_ptr=lengths,
        num_segments=num_segments,
        BLOCK_SIZE=BLOCK_SIZE,
    )
