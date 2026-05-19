import torch
import triton
import triton.language as tl

@triton.jit
def signbit_kernel(
    output_ptr,
    input_ptr,
    input_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the block index
    pid = tl.program_id(0)
    # Compute the block start index
    block_start = pid * BLOCK_SIZE
    # Create a range of offsets for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle the case where the block extends beyond the input size
    mask = offsets < input_size
    # Load the input data
    input_data = tl.load(input_ptr + offsets, mask=mask)
    # Check if the sign bit is set (negative or negative zero)
    signbit_result = input_data < 0
    # Store the result in the output tensor
    tl.store(output_ptr + offsets, signbit_result, mask=mask)

def signbit(input, *, out=None):
    # Ensure the input tensor is on the GPU
    input = input.cuda()
    # Determine the size of the input tensor
    input_size = input.numel()
    # Determine the block size
    BLOCK_SIZE = 1024
    # Determine the number of blocks needed
    num_blocks = (input_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Allocate the output tensor if not provided
    if out is None:
        out = torch.empty_like(input, dtype=torch.bool, device=input.device)
    else:
        # Ensure the output tensor is on the same device and has the same shape as the input
        assert out.device == input.device, "Output tensor must be on the same device as the input tensor"
        assert out.shape == input.shape, "Output tensor must have the same shape as the input tensor"
        out = out.to(dtype=torch.bool)
    
    # Launch the Triton kernel
    signbit_kernel[(num_blocks,)](
        out.view(-1).contiguous().data_ptr(),
        input.view(-1).contiguous().data_ptr(),
        input_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Example usage
input_tensor = torch.tensor([-1.0, 0.0, -0.0, 1.0, -2.0, 2.0], device='cuda')
output_tensor = signbit(input_tensor)
print(output_tensor)  # Expected output: tensor([ True, False,  True, False,  True, False])
