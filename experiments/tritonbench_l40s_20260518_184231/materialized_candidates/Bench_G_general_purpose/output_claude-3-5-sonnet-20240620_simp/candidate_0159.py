import torch
import triton
import triton.language as tl

@triton.jit
def nested3_kernel(
    in_ptr, out_ptr,
    stride_n,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate base indices for 2x2 block
    base_i = (pid // BLOCK_SIZE) * 2  
    base_j = (pid % BLOCK_SIZE) * 2

    # Load 2x2 block with nested loops
    for i in range(2):
        for j in range(2):
            # Calculate current position
            curr_i = base_i + i
            curr_j = base_j + j
            
            # Calculate input and output offsets
            in_offset = curr_i * stride_n + curr_j
            out_offset = curr_i * stride_n + curr_j
            
            # Load value from input
            val = tl.load(in_ptr + in_offset)
            
            # Store to output
            tl.store(out_ptr + out_offset, val)

def wrapper_nested3(N=32):
    # Create input and output tensors
    x = torch.arange(N*N, dtype=torch.float32).reshape(N, N).cuda()
    output = torch.empty_like(x)
    
    # Calculate grid size (number of 2x2 blocks)
    grid = (N * N // 4,)  # Each thread handles a 2x2 block
    
    # Launch kernel
    nested3_kernel[grid](
        x, output,
        x.stride(0),  # stride_n
        N//2,         # BLOCK_SIZE
    )
    
    print("Input tensor:")
    print(x.cpu())
    print("\nOutput tensor:")
    print(output.cpu())
    
    return output

# Run the wrapper
if __name__ == "__main__":
    wrapper_nested3()
