import triton
import triton.language as tl
import torch

# Triton kernel for the pow operation
@triton.jit
def pow_kernel(
    output_ptr, input_ptr, exponent_ptr, input_n_elements, exponent_n_elements, 
    input_strides, exponent_strides, BLOCK_SIZE: tl.constexpr
):
    # Get the program ID
    pid = tl.program_id(0)
    # Compute the range of elements for this program
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, input_n_elements)
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load input and exponent values
    input_offsets = offsets * input_strides
    input_values = tl.load(input_ptr + input_offsets, mask=offsets < input_n_elements, other=0.0)
    
    if exponent_n_elements == 1:
        # Scalar exponent
        exponent_value = tl.load(exponent_ptr)
        output_values = tl.math.pow(input_values, exponent_value)
    else:
        # Tensor exponent
        exponent_offsets = offsets * exponent_strides
        exponent_values = tl.load(exponent_ptr + exponent_offsets, mask=offsets < exponent_n_elements, other=0.0)
        output_values = tl.math.pow(input_values, exponent_values)
    
    # Store the result
    output_offsets = offsets * input_strides
    tl.store(output_ptr + output_offsets, output_values, mask=offsets < input_n_elements)

# Wrapper function for the pow operation
def pow(input: torch.Tensor, exponent, *, out=None) -> torch.Tensor:
    # Ensure input is a tensor
    assert isinstance(input, torch.Tensor)
    
    # Handle scalar exponent
    if isinstance(exponent, (int, float)):
        exponent = torch.tensor([exponent], dtype=input.dtype, device=input.device)
    
    # Ensure exponent is a tensor
    assert isinstance(exponent, torch.Tensor)
    
    # Ensure input and exponent are on the same device
    assert input.device == exponent.device
    
    # Ensure the shapes are broadcastable
    if exponent.numel() > 1:
        assert input.shape == exponent.shape, "Input and exponent shapes must be broadcastable"
    
    # Determine the number of elements
    input_n_elements = input.numel()
    exponent_n_elements = exponent.numel()
    
    # Determine the strides
    input_strides = input.stride(0)
    exponent_strides = exponent.stride(0)
    
    # Determine the block size
    BLOCK_SIZE = triton.next_power_of_2(max(input_n_elements, exponent_n_elements))
    
    # Create the output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor shape must match input tensor shape"
    
    # Ensure the tensors are on the same device
    assert input.is_cuda and exponent.is_cuda and out.is_cuda
    
    # Launch the Triton kernel
    pow_kernel[(input_n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE](
        out, input, exponent, input_n_elements, exponent_n_elements, 
        input_strides, exponent_strides, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
