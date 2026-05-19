import triton
import triton.language as tl

@triton.jit
def pow_func_scalar_tensor_kernel_rank_1(
    val0,
    in0_ptr, out0_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    
    # Compute the start index for this tile
    start_idx = pid * BLOCK_SIZE
    
    # Create a mask to handle the case where n_elements is not a multiple of BLOCK_SIZE
    mask = tl.arange(0, BLOCK_SIZE) < n_elements - start_idx
    
    # Load input values
    x = tl.load(in0_ptr + start_idx + tl.arange(0, BLOCK_SIZE), mask=mask)
    
    # Perform the power operation
    output = tl.power(x, val0)
    
    # Store the result
    tl.store(out0_ptr + start_idx + tl.arange(0, BLOCK_SIZE), output, mask=mask)

def pow_func_scalar_tensor_wrapper_rank_1(val0, in0, out0):
    assert in0.is_contiguous(), "Input tensor must be contiguous"
    assert out0.is_contiguous(), "Output tensor must be contiguous"
    assert in0.shape == out0.shape, "Input and output tensors must have the same shape"
    
    n_elements = in0.numel()
    
    # Determine the optimal block size and grid
    BLOCK_SIZE = 1024
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Determine the number of warps
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    elif BLOCK_SIZE >= 1024:
        num_warps = 4
    elif BLOCK_SIZE >= 512:
        num_warps = 2
    
    # Launch the kernel
    pow_func_scalar_tensor_kernel_rank_1[grid](
        val0,
        in0, out0,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

# Example usage
if __name__ == "__main__":
    import torch
    
    # Create input tensor
    in0 = torch.randn(1000000, device='cuda')
    
    # Create output tensor
    out0 = torch.empty_like(in0)
    
    # Set scalar value
    val0 = 2.5
    
    # Call the wrapper function
    pow_func_scalar_tensor_wrapper_rank_1(val0, in0, out0)
    
    # Verify the result
    expected_output = torch.pow(in0, val0)
    assert torch.allclose(out0, expected_output, rtol=1e-3, atol=1e-3)
    print("Triton kernel output matches PyTorch implementation.")
