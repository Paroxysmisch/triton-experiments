import triton
import triton.language as tl

@triton.jit
def isfinite_func_kernel_rank_1(
    input_ptr,  # *pointer* to the input tensor
    output_ptr,  # *pointer* to the output tensor
    n_elements,  # number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # block size
):
    # Compute the block index
    pid = tl.program_id(axis=0)
    # Compute the block start index
    block_start = pid * BLOCK_SIZE
    # Compute the block end index
    block_end = block_start + BLOCK_SIZE
    # Create a block pointer for the input and output
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(block_start,),
        block_shape=(BLOCK_SIZE,),
        order=(0,)
    )
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(block_start,),
        block_shape=(BLOCK_SIZE,),
        order=(0,)
    )
    # Load the input block
    input_block = tl.load(input_block_ptr, mask=block_start + tl.arange(0, BLOCK_SIZE) < n_elements)
    # Apply the isfinite function
    output_block = tl.isfinite(input_block)
    # Store the output block
    tl.store(output_block_ptr, output_block, mask=block_start + tl.arange(0, BLOCK_SIZE) < n_elements)

import torch
import triton
import triton.language as tl

def isfinite_func_wrapper_rank_1(input_tensor: torch.Tensor, output_tensor: torch.Tensor):
    # Ensure the input and output tensors are on the same device
    assert input_tensor.device == output_tensor.device, "Input and output tensors must be on the same device"
    assert input_tensor.dtype in [torch.float32, torch.float64], "Input tensor must be of type float32 or float64"
    assert output_tensor.dtype == torch.bool, "Output tensor must be of type bool"
    
    # Get the number of elements in the tensor
    n_elements = input_tensor.numel()
    
    # Determine the optimal block size
    BLOCK_SIZE = 1024  # This is a common choice, but you can tune it for your specific hardware
    
    # Determine the number of blocks
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    isfinite_func_kernel_rank_1[(num_blocks,)](
        input_tensor,  # *pointer* to the input tensor
        output_tensor,  # *pointer* to the output tensor
        n_elements,  # number of elements in the tensor
        BLOCK_SIZE,  # block size
    )

# Example usage
if __name__ == "__main__":
    # Create a random input tensor
    input_tensor = torch.randn(1024, device="cuda", dtype=torch.float32)
    # Create an output tensor of the same shape with boolean type
    output_tensor = torch.empty_like(input_tensor, dtype=torch.bool, device="cuda")
    
    # Call the wrapper function
    isfinite_func_wrapper_rank_1(input_tensor, output_tensor)
    
    # Print the result
    print(output_tensor)
