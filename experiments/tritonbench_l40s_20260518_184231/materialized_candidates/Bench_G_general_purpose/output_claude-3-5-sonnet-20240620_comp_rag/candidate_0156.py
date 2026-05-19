import torch
import triton
import triton.language as tl

@triton.jit
def nested3(
    in_ptr,  # Pointer to input tensor
    out_ptr, # Pointer to output tensor
    stride_m, # Stride for rows
    stride_n, # Stride for columns
    BLOCK_SIZE: tl.constexpr = 2  # Static block size
):
    # Create offset arrays for 2x2 blocks
    offs_am = tl.arange(0, BLOCK_SIZE)
    offs_an = tl.arange(0, BLOCK_SIZE)
    
    # Calculate initial input pointers for the block
    a_ptrs = in_ptr + (offs_am[:, None] * stride_m + offs_an[None, :] * stride_n)
    
    # Calculate initial output pointers for the block
    offs_cm = tl.arange(0, BLOCK_SIZE)
    offs_cn = tl.arange(0, BLOCK_SIZE)
    c_ptrs = out_ptr + (offs_cm[:, None] * stride_m + offs_cn[None, :] * stride_n)
    
    # First nested loop
    for i in range(BLOCK_SIZE):
        # Load first block
        a1 = tl.load(a_ptrs)
        
        # Second nested loop
        for j in range(BLOCK_SIZE):
            # Update pointers and load second block
            a_ptrs += BLOCK_SIZE * stride_n
            a2 = tl.load(a_ptrs)
            
            # Third nested loop
            for k in range(BLOCK_SIZE):
                # Update pointers and load third block
                a_ptrs += BLOCK_SIZE * stride_n
                a3 = tl.load(a_ptrs)
                
                # Store the loaded values
                tl.store(c_ptrs, a1)
                c_ptrs += BLOCK_SIZE * stride_n
                
                tl.store(c_ptrs, a2)
                c_ptrs += BLOCK_SIZE * stride_n
                
                tl.store(c_ptrs, a3)
                c_ptrs += BLOCK_SIZE * stride_n
        
        # Update input pointer for next iteration
        a_ptrs += BLOCK_SIZE * stride_n

def wrapper_nested3(n_rows: int, n_cols: int):
    # Create input tensor
    x = torch.arange(0, n_rows * n_cols, device="cuda", dtype=torch.float32).reshape([n_rows, n_cols])
    
    # Create output tensor
    output = torch.zeros([n_rows, n_cols], device="cuda", dtype=torch.float32)
    
    # Define grid configuration
    grid = lambda meta: (n_cols // 4,)
    
    # Launch kernel
    nested3[grid](
        x,
        output,
        x.stride(0),
        x.stride(1)
    )
    
    return output

# Example usage
if __name__ == "__main__":
    n_rows, n_cols = 4, 48
    result = wrapper_nested3(n_rows, n_cols)
    print(result)
