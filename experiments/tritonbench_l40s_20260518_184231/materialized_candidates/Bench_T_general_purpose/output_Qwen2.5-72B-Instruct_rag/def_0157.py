import triton
import triton.language as tl
import torch
import math

# Kernel for signbit operation
@triton.jit
def signbit_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    A kernel to compute the sign bit of each element in the input tensor.
    
    Parameters:
    - input_ptr: Pointer to the input tensor.
    - output_ptr: Pointer to the output tensor.
    - n_elements: Total number of elements in the tensor.
    - BLOCK_SIZE: The block size for Triton kernel execution.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = input < 0
    tl.store(output_ptr + offsets, output, mask=mask)

# Kernel for bitwise AND operation
@triton.jit
def bitwise_and_kernel(input_ptr, other_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    A kernel to perform bitwise AND operation on two tensors.
    
    Parameters:
    - input_ptr: Pointer to the input tensor.
    - other_ptr: Pointer to the other tensor.
    - output_ptr: Pointer to the output tensor.
    - n_elements: Total number of elements in the tensors.
    - BLOCK_SIZE: The block size for Triton kernel execution.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    other = tl.load(other_ptr + offsets, mask=mask)
    output = input & other
    tl.store(output_ptr + offsets, output, mask=mask)

# Wrapper function to perform signbit and bitwise AND operations
def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the sign bit check and the bitwise AND operation on the input tensors.
    
    Args:
        input (Tensor): The input tensor.
        other (Tensor): The second tensor for bitwise AND, should be of integral or boolean types.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the results of the signbit and bitwise AND operations.
    """
    # Ensure the input and other tensors are on the same device
    device = input.device
    assert other.device == device, "Input and other tensors must be on the same device"
    
    # Ensure the other tensor is of integral or boolean type
    assert other.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool], \
        "The other tensor must be of integral or boolean type"
    
    # Create output tensors
    signbit_result = torch.empty_like(input, dtype=torch.bool, device=device)
    bitwise_and_result = torch.empty_like(input, dtype=other.dtype, device=device)
    
    # Number of elements
    n_elements = input.numel()
    
    # Block size
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    
    # Grid size
    grid_size = triton.cdiv(n_elements, block_size)
    
    # Launch signbit kernel
    signbit_kernel[(grid_size, 1, 1)](input, signbit_result, n_elements, block_size)
    
    # Launch bitwise AND kernel
    bitwise_and_kernel[(grid_size, 1, 1)](input, other, bitwise_and_result, n_elements, block_size)
    
    return signbit_result, bitwise_and_result

# Example usage
if __name__ == "__main__":
    a = torch.tensor([0.7, -1.2, 0., 2.3], device='cuda')
    b = torch.tensor([1, 0, 1, 1], dtype=torch.int8, device='cuda')
    signbit_result, bitwise_and_result = signbit_bitwise_and(a, b)
    print("Signbit result:", signbit_result)
    print("Bitwise AND result:", bitwise_and_result)
