import torch
import triton
import triton.language as tl
from triton.runtime.driver import CPUDriver

@triton.jit
def nested3(
    in_ptr,  # Pointer to input tensor
    out_ptr, # Pointer to output tensor 
    stride_m, # Stride for rows
    stride_n  # Stride for columns
):
    # Create 2x2 offset grids for input
    offs_am = tl.arange(0, 2)
    offs_an = tl.arange(0, 2) 
    
    # Calculate input pointers for 2x2 block
    a_ptrs = in_ptr + (offs_am[:, None] * stride_m + 
                       offs_an[None, :] * stride_n)
    
    # Create 2x2 offset grids for output
    offs_cm = tl.arange(0, 2)
    offs_cn = tl.arange(0, 2)
    
    # Calculate output pointers for 2x2 block
    c_ptrs = out_ptr + (offs_cm[:, None] * stride_m + 
                        offs_cn[None, :] * stride_n)

    # Nested loops to process 2x2 blocks
    for i in range(2):
        # Load first 2x2 block
        a1 = tl.load(a_ptrs)
        
        for j in range(2):
            # Shift and load second 2x2 block
            a_ptrs += 2 * stride_n
            a2 = tl.load(a_ptrs)
            
            for k in range(2):
                # Shift and load third 2x2 block
                a_ptrs += 2 * stride_n
                a3 = tl.load(a_ptrs)
                
                # Store the three loaded blocks sequentially
                tl.store(c_ptrs, a1)
                c_ptrs += 2 * stride_n
                
                tl.store(c_ptrs, a2)
                c_ptrs += 2 * stride_n
                
                tl.store(c_ptrs, a3)
                c_ptrs += 2 * stride_n
        
        # Shift input pointer for next iteration
        a_ptrs += 2 * stride_n

def wrapper_nested3():
    # Initialize input dimensions
    n_rows = 4
    n_cols = 48
    
    # Create input tensor
    x = torch.arange(0, n_rows * n_cols, device="cpu", dtype=torch.int32).reshape([n_rows, n_cols])
    
    # Create output tensor
    output = torch.zeros([n_rows, n_cols], device=x.device, dtype=x.dtype)
    
    # Configure execution grid (n_cols // 4 blocks)
    grid = lambda meta: (n_cols // 4,)
    
    print("Input tensor:")
    print(x)
    print("\nInitial output tensor:")
    print(output)
    
    # Launch kernel
    nested3[grid](x, output, x.stride(0), x.stride(1))
    
    print("\nFinal output tensor:")
    print(output)
    
    return output

# Run the wrapper function
if __name__ == "__main__":
    wrapper_nested3()
