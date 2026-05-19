import torch
import triton
import triton.language as tl

@triton.jit
def nested3(
    in_ptr, out_ptr,
    stride_m, stride_n,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Base offsets for the block
    base_m = pid * BLOCK_SIZE
    
    # Loop bounds for nested iterations
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # Calculate offsets for current iteration
                m_offset = base_m + i * stride_m
                n_offset = j * stride_n + k
                
                # Calculate input and output pointers
                a_ptrs = in_ptr + m_offset + n_offset
                c_ptrs = out_ptr + m_offset + n_offset
                
                # Load values from input
                a = tl.load(a_ptrs)
                
                # Store values to output
                tl.store(c_ptrs, a)

def wrapper_nested3(n_rows, n_cols):
    # Create input tensor
    x = torch.arange(n_rows * n_cols, dtype=torch.float32, device='cuda').reshape(n_rows, n_cols)
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Calculate strides
    stride_m = n_cols
    stride_n = 1
    
    # Define grid size (assuming BLOCK_SIZE=4)
    BLOCK_SIZE = 4
    grid = (n_cols // BLOCK_SIZE,)
    
    # Launch kernel
    nested3[grid](
        x, output,
        stride_m, stride_n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Example usage
if __name__ == "__main__":
    n_rows = 8
    n_cols = 8
    result = wrapper_nested3(n_rows, n_cols)
    print(result)
