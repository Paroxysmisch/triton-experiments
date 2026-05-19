import triton
import triton.language as tl
import torch

@triton.jit
def pow_func_scalar_tensor_kernel_rank_1(
    val0, 
    in0_ptr, 
    out0_ptr, 
    N, 
    BLOCK_SIZE: tl.constexpr
):
    # Get the program ID for this kernel
    pid = tl.program_id(axis=0)
    
    # Calculate the start and end indices for this block
    start = pid * BLOCK_SIZE
    end = tl.min(start + BLOCK_SIZE, N)
    
    # Iterate over the block
    for i in range(start, end):
        # Load input value
        in_val = tl.load(in0_ptr + i)
        
        # Compute power
        result = tl.pow(in_val, val0)
        
        # Store result
        tl.store(out0_ptr + i, result)

def pow_func_scalar_tensor_wrapper_rank_1(val0, in0, out0):
    # Ensure input is a contiguous tensor
    in0 = in0.contiguous()
    
    # Get the number of elements in the input tensor
    N = in0.numel()
    
    # Determine optimal block size
    BLOCK_SIZE = 128  # You can use heuristics to determine this
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Launch the kernel
    pow_func_scalar_tensor_kernel_rank_1[grid](
        val0, 
        in0.data_ptr(), 
        out0.data_ptr(), 
        N, 
        BLOCK_SIZE
    )

# Example usage
val0 = 2.0  # Scalar value
in0 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)  # Input tensor
out0 = torch.empty_like(in0)  # Output tensor

# Call the wrapper function
pow_func_scalar_tensor_wrapper_rank_1(val0, in0, out0)

print(out0)  # Should print tensor([1.0, 4.0, 9.0, 16.0])
